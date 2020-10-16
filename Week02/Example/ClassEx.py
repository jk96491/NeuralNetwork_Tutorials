
class Rectangle:
    def __init__(self, width, height):
        print("클래스 초기화")
        self.width = width
        self.height = height

    def SetWidth(self, width):
        self.width = width

    def SetHeight(self, height):
        self.height = height

    def GetArea(self):
        area = self.width * self.height
        return area

    def GetRound(self):
        round = (self.width + self.height) * 2
        return round


rec = Rectangle(10, 20)

print("가로 : {0}, 세로 : {1}".format(rec.width, rec.height))
print("넓이 : {0}".format(rec.GetArea()))
print("둘레 : {0}".format(rec.GetRound()))






