"""

node editor

pip install NodeGraphQt
pip install PySide6
"""


from NodeGraphQt import NodeGraph, BaseNode
from PySide6 import QtWidgets, QtGui
from typing import cast
from PySide6.QtWidgets import QWidget


from typing import Callable, Optional


class MyNode(BaseNode):
    __identifier__ = 'example'
    NODE_NAME = 'MyNode'

    def __init__(self):
        super().__init__()
        self.add_input('in')
        self.add_output('out')


class MyNodeEditor:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.app.setWindowIcon(QtGui.QIcon("node_editor_qt.icon.png"))  # icon (top left)

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Node Graph 1.0")
        self.window.resize(900, 600)

        # Create the graph
        self.graph = NodeGraph()
        # self.window.setCentralWidget(self.graph.widget)
        self.window.setCentralWidget(cast(QWidget, self.graph.widget))  # cast for pylance

        # Register nodes
        self.graph.register_node(MyNode)

        # Build UI
        self._create_menu_bar()
        self._populate_graph()

    # ================================================================
    # [Menu]
    # ----------------------------------------------------------------

    filename = "node_editor_qt.save.json"

    def on_new(self):
        print("🐌 on_new()...")

    def on_open(self):
        print("🐌 on_open()...")
        self.graph.load_session(self.filename)

    def on_save(self):
        print("🐌 on_save()...")
        self.graph.save_session(self.filename)

    def on_save_as(self):
        print("🐌 on_save_as()...")

    def on_delete(self):
        nodes = self.graph.selected_nodes()
        if nodes:
            self.graph.delete_nodes(nodes)

    def _create_menu_bar(self):
        menu_bar = self.window.menuBar()

        def add_menu(name: str) -> QtWidgets.QMenu:
            return menu_bar.addMenu(name)

        def add_menu_entry(
                menu: QtWidgets.QMenu,
                name: str,
                callback: Callable | None = None,
                shortcut: str = ""
        ):

            action = QtGui.QAction(name, self.window)
            if callback:
                action.triggered.connect(callback)
            action.setShortcut(shortcut)
            menu.addAction(action)

        # clear out the default shortcuts (we will set undo/redo back)
        for a in self.window.findChildren(QtGui.QAction):
            a.setShortcuts([])

        # [File]
        file_menu = add_menu("File")
        add_menu_entry(file_menu, "New", self.on_new, "Ctrl+N")
        add_menu_entry(file_menu, "Open", self.on_open, "Ctrl+O")
        file_menu.addSeparator()
        add_menu_entry(file_menu, "Save", self.on_save, "Ctrl+S")
        add_menu_entry(file_menu, "Save As", self.on_save_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        add_menu_entry(file_menu, "Quit", self.window.close, "Ctrl+Q")

        # [Edit]
        edit_menu = add_menu("Edit")
        add_menu_entry(edit_menu, "Undo", self.graph.undo_stack().undo, "Ctrl+Z")
        add_menu_entry(edit_menu, "Redo", self.graph.undo_stack().redo, "Ctrl+Shift+Z")
        edit_menu.addSeparator()
        add_menu_entry(edit_menu, "Copy", self.graph.copy_nodes, "Ctrl+C")
        add_menu_entry(edit_menu, "Paste", self.graph.paste_nodes, "Ctrl+V")
        add_menu_entry(edit_menu, "Delete", self.on_delete, "Delete")
        add_menu_entry(edit_menu, "Select All", self.graph.select_all, "Ctrl+A")

        # [Nodes]
        node_menu = add_menu("Nodes")
        add_menu_entry(node_menu, "Add Node", self._add_node)

    # ================================================================
    # [Graph Population]
    # ----------------------------------------------------------------

    def _populate_graph(self):
        node_a = self.graph.create_node('example.MyNode', name='Node A', pos=(0, 0))
        node_b = self.graph.create_node('example.MyNode', name='Node B', pos=(250, 0))

        # Universal port connection API
        node_a.output_ports()[0].connect_to(node_b.input_ports()[0])

    def _add_node(self):
        self.graph.create_node('example.MyNode', name='New Node', pos=(100, 100))

    # ================================================================
    def run(self):

        print("🐌 load editor")

        self.window.show()
        self.app.exec()

        print("🐌 finish editor")


if __name__ == '__main__':
    editor = MyNodeEditor()
    editor.run()
