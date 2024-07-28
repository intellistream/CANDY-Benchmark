from typing import List
import random
class IntelliTimeStamp:
    def __init__(self, eventTime=0, arrivalTime=0, processedTime=0):
        self.eventTime = eventTime
        self.arrivalTime = arrivalTime
        self.processedTime = processedTime
    
    def __repr__(self):
        return f"IntelliTimeStamp(eventTime={self.eventTime}, arrivalTime={self.arrivalTime}, processedTime={self.processedTime}"

def generate_random_list(length: int, maxTime: int, minTime=0) -> List[int]:
    return [random.randint(0, maxTime - 1) for _ in range(length)]

def gen_smooth_timestamp(length: int, maxTime: int) -> List[int]:
    ret = generate_random_list(length, maxTime)
    ret.sort()  # Incremental rearrangement (sort the list)
    return ret

class IntelliTimeStampGenerator:
    def __init__(self, testSize=0):
        self.testSize = testSize
        self.eventS: List[int] = []
        self.arrivalS: List[int] = []
        self.eventRateTps = 100
        self.timeStepUs = 40
        self.seed = 114514
        self.staticDataSet = 0
        self.myTs: List[IntelliTimeStamp] = []
        self.generateEvent()
        self.generateArrival()
        self.generateFinal()

    def generateEvent(self):
        # Implementation for generating the vector of events
        
        maxTime = self.testSize * 1000 * 1000 // self.eventRateTps
        self.eventS = gen_smooth_timestamp(self.testSize, maxTime);  # Create a list of size 'testSize'
        print("Finish the generation of event time")
        return

    def generateArrival(self):
        # Implementation for generating the vector of arrivals
        pass

    def generateFinal(self):
        # Implementation for generating the final result of s and r
        self.myTs=self.constructTimeStamps(self.eventS, self.eventS)
        return

    def constructTimeStamps(self, eventS: List[int], arrivalS: List[int]) -> List[IntelliTimeStamp]:
        
        ru = [IntelliTimeStamp(event, arrival, 0) for event, arrival in zip(eventS, arrivalS)]
        return ru

    def setConfig(self, cfg: dict) -> bool:
        # Implementation for setting the global config map related to this TimerStamper
        return True

    def getTimeStamps(self) -> List[IntelliTimeStamp]:
        # Implementation for getting the vector of time stamps
        return self.myTs
    
tsg = IntelliTimeStampGenerator(50)
print(tsg.arrivalS)
print(tsg.eventS)
print(tsg.myTs)