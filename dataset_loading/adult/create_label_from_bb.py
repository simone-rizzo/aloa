import pandas as pd
from sklearn.metrics import classification_report
import bboxes
from bboxes.dtbb import DecisionTreeBlackBox
from bboxes.rfbb import RandomForestBlackBox

# bb = DecisionTreeBlackBox()
bb = RandomForestBlackBox()

noise_data = pd.read_csv("data/noise_shadow.csv")
print(noise_data.shape)
predictions = bb.predict(noise_data.values)
noise_data['class'] = predictions
noise_data.to_csv("data/adult_noise_shadow_labelled", index=False)