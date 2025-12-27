from PySide6 import QtWidgets, QtGui, QtCore

app = QtWidgets.QApplication([])

scene = QtWidgets.QGraphicsScene()
scene.setSceneRect(0, 0, 400, 300)
scene.addRect(0, 0, 200, 100, QtGui.QPen(QtGui.QColor("red")), QtGui.QBrush(QtGui.QColor("yellow")))

view = QtWidgets.QGraphicsView(scene)
view.resize(400, 300)
view.show()

app.exec()
