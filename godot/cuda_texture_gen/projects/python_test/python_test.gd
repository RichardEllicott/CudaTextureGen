"""
"""
#@tool
extends Control


@onready var py := $PythonBridge

func _ready():
    print("Sending test line...")
    py.SendLine("hello from godot")


func _process(delta):
    #print("ss")
    if py.IsRunning():
        print("ss")
        var line = py.ReadLine()
        if line != "":
            print("PYTHON:", line)
