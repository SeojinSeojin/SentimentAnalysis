import pandas as pd
from sklearn.model_selection import train_test_split

# CSV 읽기
df = pd.read_csv("all.csv", header=0, encoding="cp1252")

print(df.shape)
print(df.columns.tolist())

# 13번째 컬럼(index 12)이 1인 것만 필터
col_relevant = df.columns[12]
col_sentiment = df.columns[13]
col_title = "title"
col_body = "body"

df_filtered = df[df[col_relevant].astype(str).str.strip().isin(["1", "0"])].copy()
df_filtered = df_filtered.dropna(subset=[col_sentiment])

print(df_filtered.shape)

# title + body 합치기
df_filtered["sentence"] = df_filtered[col_title].fillna("") + " " + df_filtered[col_body].fillna("")

# sentence랑 sentiment만 추출
df_out = df_filtered[["sentence", col_sentiment]].copy()

df_out.columns = ["sentence", "label"]
df_out = df_out[df_out["label"].isin(["-1", "0", "1"])]

# label 값 확인
print("Label 분포:")
print(df_out["label"].value_counts())
print(f"총 {len(df_out)}개")

# Positive/Negative/Neutral 비율 유지하면서 8:2로 나누기
train_df, dev_df = train_test_split(
    df_out,
    test_size=0.2,
    random_state=42,
    stratify=df_out["label"]  # 이게 비율 유지해주는 핵심
)

# 저장
train_df.to_csv("train.tsv", sep="\t", index=False)
dev_df.to_csv("dev.tsv", sep="\t", index=False)

print(f"\nTrain: {len(train_df)}개")
print(f"Dev: {len(dev_df)}개")
print("\nTrain 분포:")
print(train_df["label"].value_counts())
print("\nDev 분포:")
print(dev_df["label"].value_counts())