"""

simple dearpygui based node editor

"""


import dearpygui.dearpygui as dpg


MENU_REGISTRY = {}


def menu(name: str):
    MENU_REGISTRY.setdefault(name, [])

    def wrapper(func):
        # We don't store the function here — just ensure the menu exists
        return func
    return wrapper


def menu_item(menu_name: str, label: str):
    def wrapper(func):
        MENU_REGISTRY.setdefault(menu_name, [])
        MENU_REGISTRY[menu_name].append(("item", label, func))
        return func
    return wrapper


def menu_separator(menu_name: str):
    def wrapper(func):
        MENU_REGISTRY.setdefault(menu_name, [])
        MENU_REGISTRY[menu_name].append(("separator", None, None))
        return func
    return wrapper


class NodeEditorApp:

    main_window: int
    node_editor: int

    def __init__(self):

        print("🧸 node editor starting...")

        dpg.create_context()
        dpg.create_viewport(title="Node Editor", width=800, height=600)

        self._build_ui()

        dpg.set_viewport_resize_callback(self._on_viewport_resize)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()  # main loop

        print("🧸 node editor closing...")

        dpg.destroy_context()

    # -------------------------
    # UI Construction
    # -------------------------
    def _build_ui(self):
        with dpg.window(
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            no_close=True,
            pos=(0, 0),
            width=dpg.get_viewport_width(),
            height=dpg.get_viewport_height()
        ) as self.main_window:

            self._build_menu_bar()
            self._build_node_editor()

    # -------------------------
    # Menu callbacks
    # -------------------------

    @menu("File")
    @menu_item("File", "New")
    def on_menu_new(self):
        pass

    @menu_item("File", "Open")
    def on_menu_open(self):
        pass

    @menu_separator("File")
    def _sep_file_1(self):
        pass

    @menu_item("File", "Save")
    def on_menu_save(self):
        pass
    
    @menu_item("File", "Save As")
    def on_menu_save_as(self):
        pass

    @menu_separator("File")
    def _sep_file_2(self):
        pass

    @menu_item("File", "Exit")
    def on_menu_exit(self):
        dpg.stop_dearpygui()


    @menu("Edit")
    @menu_item("Edit", "Undo")
    def on_menu_undo(self):
        pass
    
    @menu_item("Edit", "Redo")
    def on_menu_redo(self):
        pass

    def _build_menu_bar(self):
        with dpg.menu_bar():
            for menu_name, entries in MENU_REGISTRY.items():
                with dpg.menu(label=menu_name):
                    for entry_type, label, func in entries:

                        if entry_type == "item":
                            # Bind the function to this instance
                            bound = func.__get__(self, self.__class__)
                            dpg.add_menu_item(label=label, callback=bound)

                        elif entry_type == "separator":
                            dpg.add_separator()

            with dpg.menu(label="End"):


                dpg.add_menu_item(label="Option1", callback=self.on_menu_new)
            

    # def _build_menu_bar(self):

    #     with dpg.menu_bar():

    #         with dpg.menu(label="File"):
    #             dpg.add_menu_item(label="New", callback=self.on_menu_new)
    #             dpg.add_menu_item(label="Open", callback=self.on_menu_open)

    #             dpg.add_separator()

    #             dpg.add_menu_item(label="Save", callback=self.on_menu_save)
    #             dpg.add_menu_item(label="Save As", callback=self.on_menu_save_as)

    #             dpg.add_separator()

    #             dpg.add_menu_item(label="Exit", callback=self.on_menu_exit)

    #         with dpg.menu(label="Edit"):
    #             dpg.add_menu_item(label="Undo")
    #             dpg.add_menu_item(label="Redo")

    def _build_node_editor(self):
        with dpg.node_editor(
            callback=self._on_link_created,
            delink_callback=self._on_link_deleted
        ) as self.node_editor:

            # Node A
            with dpg.node(label="Node A"):
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as a_out:
                    dpg.add_text("A Output")

            # Node B
            with dpg.node(label="Node B"):
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as b_in:
                    dpg.add_text("B Input")

    # -------------------------
    # Callbacks
    # -------------------------
    def _on_link_created(self, sender, app_data):
        print("Link created:", app_data)
        dpg.add_node_link(app_data[0], app_data[1], parent=sender)

    def _on_link_deleted(self, sender, app_data):
        print("Link deleted:", app_data)
        dpg.delete_item(app_data)

    def _on_viewport_resize(self, sender, app_data):
        dpg.configure_item(
            self.main_window,
            width=dpg.get_viewport_width(),
            height=dpg.get_viewport_height()
        )


# Run it
NodeEditorApp()
