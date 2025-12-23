"""

testing dearpygui

"""
import dearpygui.dearpygui as dpg  # pip install dearpygui


dpg.create_context()
dpg.create_viewport(title="Minimal Node Editor", width=800, height=600)

# Callback when a link (wire) is created


def link_callback(sender, app_data):
    # app_data = (output_attr_id, input_attr_id)
    print("Link created:", app_data)
    dpg.add_node_link(app_data[0], app_data[1], parent=sender)

# Callback when a link is deleted


def delink_callback(sender, app_data):
    print("Link deleted:", app_data)
    dpg.delete_item(app_data)


with dpg.window(label="Node Editor Window", width=800, height=600):
    with dpg.node_editor(callback=link_callback, delink_callback=delink_callback) as editor:

        # --- Node A ---
        with dpg.node(label="Node A") as node_a:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as a_out:
                dpg.add_text("A Output")

        # --- Node B ---
        with dpg.node(label="Node B") as node_b:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as b_in:
                dpg.add_text("B Input")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
