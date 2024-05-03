import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def parse_log_entry(line):
    timestamp_str, left_pupil_str, right_pupil_str, alert_str = line.split(' - ')
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    left_pupil = None if 'None' in left_pupil_str else eval(left_pupil_str.split(': ')[1])
    right_pupil = None if 'None' in right_pupil_str else eval(right_pupil_str.split(': ')[1])
    alert = True if 'Yes' in alert_str else False
    return timestamp, left_pupil, right_pupil, alert

with open('core/logging/pupil.log', 'r') as file:
    log_entries = file.readlines()

logs = [parse_log_entry(line.strip()) for line in log_entries if line.strip()]

df = pd.DataFrame(logs, columns=['timestamp', 'left_pupil', 'right_pupil', 'alert'])

df['eye_detected'] = df['left_pupil'].notna() & df['right_pupil'].notna()
df['no_eye_period'] = (~df['eye_detected']).astype(int)
df['no_eye_period'] = df['no_eye_period'].groupby((df['no_eye_period'] != df['no_eye_period'].shift()).cumsum()).cumsum()

alerts = df[df['alert'] == True]
alert_counts = alerts['timestamp'].dt.hour.value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(df['timestamp'], df['no_eye_period'], label='No Eye Detected Duration')
plt.xlabel('Time')
plt.ylabel('Seconds')
plt.title('Duration of No Attention')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
alert_counts.plot(kind='bar')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Alerts')
plt.title('Distribution of Alerts by Hour')

plt.tight_layout()
plt.show()
