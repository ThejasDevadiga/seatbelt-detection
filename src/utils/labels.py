from enum import Enum
class PDetectionLabels(Enum):
    PERSON = 0

class ObjectDetectionLabels(Enum):
    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORBIKE = 3
    AEROPLANE = 4
    BUS = 5
    TRAIN = 6
    TRUCK = 7
    TRAFFIC_LIGHT = 9
    CELL_PHONE = 67

class SeatBeltLabels(Enum):
    no_seat_belt = 0
    seat_belt = 1

