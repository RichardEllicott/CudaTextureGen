

from PySide6 import QtCore, QtGui, QtWidgets


# -----------------------------
# Port (simple ellipse item)
# -----------------------------
class Port(QtWidgets.QGraphicsEllipseItem):
    RADIUS = 6

    def __init__(self, parent: QtWidgets.QGraphicsItem | None, is_output: bool = False):
        super().__init__(-Port.RADIUS, -Port.RADIUS,
                         Port.RADIUS * 2, Port.RADIUS * 2, parent)

        self.setBrush(QtGui.QColor("red"))  # make it obvious
        self.setPen(QtGui.QPen(QtGui.QColor("black"), 1))

        self.is_output = is_output
        self.connections: list[QtWidgets.QGraphicsPathItem] = []

    def center_pos(self) -> QtCore.QPointF:
        return self.mapToScene(self.boundingRect().center())


# -----------------------------
# Connection (Bezier path)
# -----------------------------
class Connection(QtWidgets.QGraphicsPathItem):
    def __init__(self, start_port: Port, end_port: Port | None = None):
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        self.temp_end_pos: QtCore.QPointF | None = None

        pen = QtGui.QPen(QtGui.QColor("#55AADD"), 2)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        self.setPen(pen)

    def update_path(self):
        start = self.start_port.center_pos()
        end = self.end_port.center_pos() if self.end_port else self.temp_end_pos

        if end is None:
            return

        path = QtGui.QPainterPath(start)
        dx = (end.x() - start.x()) * 0.5
        c1 = QtCore.QPointF(start.x() + dx, start.y())
        c2 = QtCore.QPointF(end.x() - dx, end.y())
        path.cubicTo(c1, c2, end)
        self.setPath(path)


# -----------------------------
# Node (rect item, no custom paint)
# -----------------------------
class Node(QtWidgets.QGraphicsRectItem):
    WIDTH = 160
    HEIGHT = 80

    def __init__(self, title: str = "Node"):
        super().__init__(0, 0, Node.WIDTH, Node.HEIGHT)

        # Move/select flags
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )

        # Appearance
        self.setBrush(QtGui.QColor("#333333"))
        self.setPen(QtGui.QPen(QtGui.QColor("#222222"), 2))

        self.title = title

        # Simple label as a child item so we don't implement paint()
        self.label = QtWidgets.QGraphicsTextItem(self.title, self)
        self.label.setDefaultTextColor(QtGui.QColor("white"))
        self.label.setPos(10, 5)

        # Ports
        self.input_port = Port(self, is_output=False)
        self.output_port = Port(self, is_output=True)

        self.input_port.setPos(0, Node.HEIGHT / 2)
        self.output_port.setPos(Node.WIDTH, Node.HEIGHT / 2)


# -----------------------------
# Scene with connection logic
# -----------------------------
class NodeScene(QtWidgets.QGraphicsScene):
    def __init__(self):
        super().__init__()

        # A modest scene rect that definitely contains our nodes
        self.setSceneRect(0, 0, 800, 600)

        self.temp_connection: Connection | None = None
        self.start_port: Port | None = None

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        item = self.itemAt(event.scenePos(), QtGui.QTransform())

        if isinstance(item, Port):
            self.start_port = item
            self.temp_connection = Connection(item)
            self.addItem(self.temp_connection)
            self.temp_connection.temp_end_pos = event.scenePos()
            self.temp_connection.update_path()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.temp_connection is not None:
            self.temp_connection.temp_end_pos = event.scenePos()
            self.temp_connection.update_path()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.temp_connection is not None:
            end_item = self.itemAt(event.scenePos(), QtGui.QTransform())

            if (
                isinstance(self.start_port, Port)
                and isinstance(end_item, Port)
                and end_item is not self.start_port
            ):
                self.temp_connection.end_port = end_item
                self.temp_connection.update_path()
                self.start_port.connections.append(self.temp_connection)
                end_item.connections.append(self.temp_connection)
            else:
                self.removeItem(self.temp_connection)

            self.temp_connection = None
            self.start_port = None

        super().mouseReleaseEvent(event)


# -----------------------------
# View (zoom + pan)
# -----------------------------
class NodeView(QtWidgets.QGraphicsView):
    def __init__(self, scene: QtWidgets.QGraphicsScene):
        super().__init__(scene)

        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # Light background so everything pops
        self.setBackgroundBrush(QtGui.QColor("#E0E0E0"))

        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        zoom = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(zoom, zoom)


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        scene = NodeScene()
        view = NodeView(scene)
        self.setCentralWidget(view)

        # Two nodes
        n1 = Node("Image Node")
        n2 = Node("Output Node")

        n1.setPos(100, 100)
        n2.setPos(400, 300)

        scene.addItem(n1)
        scene.addItem(n2)

        self.setWindowTitle("Minimal Node Editor (Rect-based)")
        self.resize(800, 600)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
