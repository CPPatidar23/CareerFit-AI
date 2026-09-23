import os, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

DATA_PATH="data/CareerFit_50000_Text_Skills_Dataset.csv"; MODEL_DIR="models"
os.makedirs(MODEL_DIR, exist_ok=True)
df=pd.read_csv(DATA_PATH).dropna(subset=["Target Career"]).copy()
df=df.drop(columns=[c for c in ["Index","Student Name"] if c in df.columns])
cols=["Skills","Interests","Favorite Subjects","Personality","Career Goal"]
for c in cols: df[c]=df[c].fillna("").astype(str)
df["Combined_Text"]=df[cols].agg(" ".join,axis=1)
X_train,X_test,y_train,y_test=train_test_split(df["Combined_Text"],df["Target Career"],test_size=.20,random_state=42,stratify=df["Target Career"])
vectorizer=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2),min_df=2,max_features=20000)
A=vectorizer.fit_transform(X_train); B=vectorizer.transform(X_test)
model=LogisticRegression(max_iter=1000,random_state=42).fit(A,y_train)
pred=model.predict(B)
print("Dataset:",df.shape); print("Training:",A.shape); print("Testing:",B.shape)
print("Accuracy:",round(accuracy_score(y_test,pred)*100,2),"%")
print(classification_report(y_test,pred))
joblib.dump(model,MODEL_DIR+"/career_model.pkl"); joblib.dump(vectorizer,MODEL_DIR+"/tfidf_vectorizer.pkl")
cm=confusion_matrix(y_test,pred,labels=model.classes_)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
fig,ax=plt.subplots(figsize=(12,10)); disp.plot(ax=ax,xticks_rotation=45,colorbar=False)
plt.tight_layout(); plt.savefig(MODEL_DIR+"/confusion_matrix.png",dpi=180); plt.close()
print("Saved model, vectorizer and confusion matrix.")
