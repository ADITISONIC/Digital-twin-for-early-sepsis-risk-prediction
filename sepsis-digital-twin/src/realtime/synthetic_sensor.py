import random
from copy import deepcopy


class SyntheticPatientSensor:

    def __init__(self, patient_id="SIM001", scenario="sepsis_deterioration"):
        self.patient_id = patient_id
        self.scenario = scenario
        self.hour = 0

        # Initial physiological state
        self.state = {
            "HR": 78.0,
            "O2Sat": 98.0,
            "Temp": 36.7,
            "SBP": 120.0,
            "MAP": 85.0,
            "DBP": 70.0,
            "Resp": 16.0,

            "Lactate": 1.2,
            "Creatinine": 0.9,
            "WBC": 8.0,
        }

    def _apply_deterioration(self):

        h = self.hour

        if self.scenario == "normal":

            # Small natural variation
            changes = {
                "HR": 0,
                "O2Sat": 0,
                "Temp": 0,
                "SBP": 0,
                "MAP": 0,
                "DBP": 0,
                "Resp": 0,
                "Lactate": 0,
                "Creatinine": 0,
                "WBC": 0,
            }

        elif self.scenario == "sepsis_deterioration":

            # Gradual deterioration
            severity = min(h / 20.0, 1.0)

            changes = {
                "HR": 35 * severity,
                "O2Sat": -7 * severity,
                "Temp": 2.0 * severity,
                "SBP": -35 * severity,
                "MAP": -25 * severity,
                "DBP": -20 * severity,
                "Resp": 12 * severity,
                "Lactate": 2.8 * severity,
                "Creatinine": 0.8 * severity,
                "WBC": 5 * severity,
            }

        elif self.scenario == "rapid_deterioration":

            severity = min(h / 10.0, 1.0)

            changes = {
                "HR": 40 * severity,
                "O2Sat": -9 * severity,
                "Temp": 2.5 * severity,
                "SBP": -45 * severity,
                "MAP": -30 * severity,
                "DBP": -25 * severity,
                "Resp": 16 * severity,
                "Lactate": 4.0 * severity,
                "Creatinine": 1.2 * severity,
                "WBC": 7 * severity,
            }

        else:
            raise ValueError(
                f"Unknown scenario: {self.scenario}"
            )

        # Apply changes
        for feature, change in changes.items():

            # Add small sensor noise
            noise = random.gauss(0, 0.3)

            self.state[feature] += (
                change / max(self.hour, 1)
            ) + noise

    def generate_reading(self):

        self._apply_deterioration()

        reading = deepcopy(self.state)

        reading["PatientID"] = self.patient_id
        reading["ICULOS"] = self.hour

        self.hour += 1

        return reading


if __name__ == "__main__":

    sensor = SyntheticPatientSensor(
        patient_id="SIM001",
        scenario="sepsis_deterioration"
    )

    for _ in range(20):

        reading = sensor.generate_reading()

        print(reading)