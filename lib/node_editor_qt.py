"""

node editor

pip install NodeGraphQt
pip install PySide6

🎨🐌

"""


import sys
import json
from NodeGraphQt import NodeGraph, BaseNode, BackdropNode
from typing import cast, Any
from PySide6.QtWidgets import QWidget, QGraphicsPixmapItem, QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox
from collections.abc import Callable
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from NodeGraphQt.qgraphics.node_base import NodeItem
from PySide6.QtCore import Qt

# these five this actual script
script_dir = Path(__file__).resolve().parent
icon_path = f"{script_dir}/node_editor_qt.icon.png"

script_filename = Path(__file__).name
scrip_stem = Path(__file__).stem  # minus the extension
# script_full_path = Path(__file__).resolve()


# this is instead the entry point
entry_script = sys.argv[0]


class PersistentSettings:
    """
    PersistentSettings saves settings to JSON
    set pars on this object and call "save" to ensure json file is updated
    """

    def __init__(self, path: str | Path, **defaults: Any) -> None:

        self._path = Path(path)
        self._synced = True

        # Start with defaults
        for k, v in defaults.items():
            setattr(self, k, v)

        # Load existing file if present
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for k, v in data.items():
                    setattr(self, k, v)
            except Exception as e:
                # You can decide whether to raise or ignore
                print(f"Warning: failed to load settings: {e}")

    def __setattr__(self, name: str, value: object) -> None:
        """
        any pars set that do not begin with underscore and have changed
        will mark the _synced as False
        indicating we need to save the json
        """

        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        # If the value is unchanged, do nothing
        if hasattr(self, name) and getattr(self, name) == value:
            return

        # Otherwise set and mark unsynced
        object.__setattr__(self, name, value)
        object.__setattr__(self, "_synced", False)

    def save(self) -> None:
        """
        will save the json if we are not already synced
        """
        if self._synced:  # pass if synced already
            return

        data = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")  # skip internal attributes
        }
        self._path.write_text(json.dumps(data, indent=2))

        self._synced = True

    def __del__(self):
        self.save()


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Substrata")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Substrata</b>"))
        layout.addWidget(QLabel("Version 1.0"))
        layout.addWidget(QLabel("Built by Richard"))
        layout.addWidget(QLabel("Using NodeGraphQt"))

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)


# to figure out the image loading:
# https://claude.ai/chat/fbdfe94e-c006-45b6-b3d8-567a3bc8e967

class CustomBaseNode(BaseNode):
    """
    Docstring for BaseNode2
    """

    __identifier__ = 'node'
    NODE_NAME = 'CustomBaseNode'

    def __init__(self):
        super().__init__()

    def set_image(self, image_path):
        """Load and set an image for this node"""
        pixmap = QtGui.QPixmap(image_path)

        if not pixmap.isNull():
            # Remove old pixmap if exists
            if self._pixmap_item:
                self.view.scene().removeItem(self._pixmap_item)

            # Scale pixmap to fit
            # scaled_pixmap = pixmap.scaled(64, 128, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                #   QtCore.Qt.TransformationMode.SmoothTransformation)

            scaled_pixmap = pixmap.scaled(64, 128, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                          QtCore.Qt.TransformationMode.SmoothTransformation)

            # Create pixmap item as child of node view
            self._pixmap_item = QGraphicsPixmapItem(scaled_pixmap)
            self._pixmap_item.setParentItem(self.view)

            # Position it within the node (adjust as needed)
            self._pixmap_item.setPos(10, 30)

            print(f"Image loaded: {image_path}")
        else:
            print(f"Failed to load image: {image_path}")


class GraphNode(CustomBaseNode):
    __identifier__ = 'node'
    NODE_NAME = 'GraphNode'

    def __init__(self):
        super().__init__()
        self.add_input('in')
        self.add_output('out')


class ImageNode(CustomBaseNode):
    """Custom node that displays an image"""

    __identifier__ = 'node'
    NODE_NAME = 'ImageNode'

    def __init__(self):
        super().__init__()

        self.add_input('in')
        # self.add_output('out')

        # Store reference to the pixmap item
        self._pixmap_item = None

        self.set_image(icon_path)  # not working with spacing

        # self.add_text_input('spacer', 'Spacer', text=' ' * 50)  # Wide text
        # self.add_button('test', 'test123')

        # self.add_label('image_label')

    def post_init(self, viewer=None, pos=None):
        """Called after node is added to the graph"""
        # super().post_init(viewer, pos)

        # Set the internal size properties
        self.view._width = 500
        self.view._height = 150

        # Recalculate the node's layout
        self.view.calc_size()

        # Force a visual update
        self.view.update()


class NodeEditorSettings(PersistentSettings):

    pass


class MyNodeEditor:

    app_name: str = "Substrata 1.0.0"
    app_version: str = "1.0.0"

    save_path: str
    autoload: bool = True

    def save_settings(self) -> None:
        """
        save settings to json file
        """
        self.settings.save_path = self.save_path
        self.settings.save()

    def load_settings(self) -> None:
        """
        load settings from json file
        """
        # some defaults, change later
        script_dir = Path(__file__).resolve().parent
        script_stem = Path(__file__).stem

        self.settings: NodeEditorSettings = NodeEditorSettings(
            self.settings_path,
            save_path=script_dir / f"{script_stem}.save.json"
        )

        save_path = getattr(self.settings, "save_path", None)
        if save_path is not None:
            self.save_path = save_path

    def save_session(self) -> None:
        session_dict = self.graph.serialize_session()
        viewer = self.graph.viewer()

        # viewport center in widget coordinates
        viewport_center = viewer.viewport().rect().center()
        # convert to scene coordinates
        scene_center = viewer.mapToScene(viewport_center)

        session_dict["meta"] = {
            "zoom": viewer.get_zoom(),
            "center": {
                "x": scene_center.x(),
                "y": scene_center.y(),
            },
            "app_version": self.app_version,
        }

        with open(self.save_path, "w") as f:
            json.dump(session_dict, f, indent=4)

    def load_session(self) -> None:
        print("🐌 load_session()...")

        # 1. Load JSON
        with open(self.save_path, "r") as f:
            session_dict = json.load(f)

        # 2. Deserialize the graph
        self.graph.deserialize_session(session_dict)

        # 3. Restore metadata if present
        meta = session_dict.get("meta", {})
        viewer = self.graph.viewer()

        # --- Restore zoom ---
        zoom = meta.get("zoom")
        if zoom is not None:
            try:
                viewer.set_zoom(zoom)
            except Exception:
                pass  # some versions use viewer.setZoom()

        # --- Restore view center ---
        center = meta.get("center")
        if center:
            cx = center.get("x", 0)
            cy = center.get("y", 0)

            # centerOn expects scene coordinates
            try:
                viewer.centerOn(cx, cy)
            except Exception:
                pass

        print("🐌 finish load_session")

    # ================================================================
    # [Menu]
    # ----------------------------------------------------------------

    def on_new(self) -> None:
        print("🐌 on_new()...")

        # 1. Delete all nodes
        for node in self.graph.all_nodes():
            self.graph.delete_node(node)

        # 2. Clear the scene (connections, backdrops, etc.)
        self.graph.clear_session()

        # 3. Reset undo stack
        self.graph.undo_stack().clear()

        # 4. Reset view (optional)
        self.graph.reset_zoom()
        self.graph.viewer().centerOn(0, 0)

    def on_open(self) -> None:
        print("🐌 on_open()...")
        # self.graph.load_session(self.save_path)
        self.load_session()

    def on_save(self) -> None:
        print("🐌 on_save()...")
        # self.graph.save_session(self.save_path)
        self.save_session()

    def on_save_as(self) -> None:
        print("🐌 on_save_as()...")
    # ----------------------------------------------------------------

    def on_delete(self) -> None:
        """
        deleted selected nodes
        """
        nodes = self.graph.selected_nodes()
        if nodes:
            self.graph.delete_nodes(nodes)
    # ----------------------------------------------------------------

    def on_reset_view(self) -> None:
        self.graph.reset_zoom()

    # ----------------------------------------------------------------

    def on_about(self) -> None:
        print("🐌 on_about()...")

        dlg = AboutDialog(self.window)
        dlg.exec()
    # ----------------------------------------------------------------

    def _create_menu_bar(self) -> None:
        menu_bar = self.window.menuBar()

        def add_menu(name: str) -> QtWidgets.QMenu:
            return menu_bar.addMenu(name)

        def add_menu_entry(
                menu: QtWidgets.QMenu,
                name: str,
                callback: Callable | None = None,
                shortcut: str = ""
        ) -> None:

            action = QtGui.QAction(name, self.window)
            if callback:
                action.triggered.connect(callback)
            action.setShortcut(shortcut)
            menu.addAction(action)

        # clear out the default shortcuts (we will set undo/redo back)
        for a in self.window.findChildren(QtGui.QAction):
            a.setShortcuts([])

        # ----------------------------------------------------------------
        # [File]
        menu = add_menu("File")
        add_menu_entry(menu, "New", self.on_new, "Ctrl+N")
        add_menu_entry(menu, "Open", self.on_open, "Ctrl+O")
        menu.addSeparator()
        add_menu_entry(menu, "Save", self.on_save, "Ctrl+S")
        add_menu_entry(menu, "Save As", self.on_save_as, "Ctrl+Shift+S")
        menu.addSeparator()
        add_menu_entry(menu, "Quit", self.window.close, "Ctrl+Q")
        # ----------------------------------------------------------------
        # [Edit]
        menu = add_menu("Edit")
        add_menu_entry(menu, "Undo", self.graph.undo_stack().undo, "Ctrl+Z")
        add_menu_entry(menu, "Redo", self.graph.undo_stack().redo, "Ctrl+Shift+Z")
        menu.addSeparator()
        add_menu_entry(menu, "Cut", self.graph.cut_nodes, "Ctrl+X")
        add_menu_entry(menu, "Copy", self.graph.copy_nodes, "Ctrl+C")
        add_menu_entry(menu, "Paste", self.graph.paste_nodes, "Ctrl+V")
        add_menu_entry(menu, "Delete", self.on_delete, "Delete")
        add_menu_entry(menu, "Select All", self.graph.select_all, "Ctrl+A")
        menu.addSeparator()
        add_menu_entry(menu, "Group", self.create_group, "Ctrl+G")
        # ----------------------------------------------------------------
        # [Nodes]
        menu = add_menu("Nodes")
        add_menu_entry(menu, "Add Node", self._add_node)
        add_menu_entry(menu, "Add Node 2", self._add_node2)
        # ----------------------------------------------------------------
        # [View]
        menu = add_menu("View")
        add_menu_entry(menu, "Reset View", self.on_reset_view)
        # ----------------------------------------------------------------
        # [Window]
        # menu = add_menu("Window")
        # ----------------------------------------------------------------
        # [About]
        menu = add_menu("Help")
        add_menu_entry(menu, "About", self.on_about)

    # ================================================================
    # [Graph Population]
    # ----------------------------------------------------------------

    def _populate_graph(self) -> None:
        node_a = self.graph.create_node('node.GraphNode', name='Node A', pos=(0, 0))
        node_b = self.graph.create_node('node.GraphNode', name='Node B', pos=(250, 0))

        # Universal port connection API
        node_a.output_ports()[0].connect_to(node_b.input_ports()[0])

    def _add_node(self) -> None:
        # self.graph.create_node('example.MyNode', name='New Node', pos=(100, 100))
        self.graph.create_node('node.GraphNode', pos=(100, 100))

    def _add_node2(self) -> None:
        # node = self.graph.create_node('custom.ImageNode', name='Image Node', pos=(100, 100))
        node = self.graph.create_node('node.ImageNode', pos=(100, 100))
        # node.set_image(icon_path)

    # ================================================================
    # ⚠️ not working

    def create_group(self) -> None:
        # get selected nodes
        nodes = self.graph.selected_nodes()
        if not nodes:
            return

        # create a backdrop
        backdrop = self.graph.create_node(BackdropNode)

        # auto-fit around selected nodes
        backdrop.fit_to_nodes(nodes)

        # optional: give it a title or color
        backdrop.set_property('name', 'My Group')
        backdrop.set_property('color', (50, 50, 80))

    # ================================================================

    def __init__(
            self,
            settings_path: str
    ) -> None:

        self.settings_path = settings_path
        self.load_settings()

        self.app = QtWidgets.QApplication([])
        self.app.setWindowIcon(QtGui.QIcon(icon_path))  # icon (top left)

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle(self.app_name)
        self.window.resize(900, 600)

        # Create the graph
        self.graph = NodeGraph()
        # self.window.setCentralWidget(self.graph.widget)
        self.window.setCentralWidget(cast(QWidget, self.graph.widget))  # cast for pylance

        # Register nodes
        self.graph.register_node(GraphNode)
        self.graph.register_node(ImageNode)

        # Build UI
        self._create_menu_bar()
        self._populate_graph()

        if self.autoload:
            self.on_open()

    def run(self) -> None:

        print("🐌 load editor")

        self.window.show()
        self.app.exec()

        self.save_settings()
        if self.autoload:
            self.on_save()

        print("🐌 finish editor")


if __name__ == '__main__':
    editor = MyNodeEditor(f"{script_dir}/{scrip_stem}.settings.json")
    editor.run()
