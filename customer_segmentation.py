import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("customer_data.csv")

model = KMeans(n_clusters=3, random_state=42)
data["Segment"] = model.fit_predict(data[["Income", "Spending"]])

print(data)
