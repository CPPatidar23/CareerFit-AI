import os, pandas as pd, streamlit as st, joblib
from expert_system import reasons, rule_score, info
st.set_page_config(page_title="CareerFit",page_icon="🎯",layout="wide")
st.title("🎯 CareerFit")
st.subheader("AI-Powered Career Guidance Expert System")
st.write("Find the career that fits your skills, interests, subjects, personality and goals.")
if not os.path.exists("models/career_model.pkl"): st.error("Run: python train_model.py"); st.stop()
model=joblib.load("models/career_model.pkl"); vec=joblib.load("models/tfidf_vectorizer.pkl")
with st.form("form"):
    a,b=st.columns(2)
    with a:
        education=st.selectbox("Education",["B.Tech","BCA","B.Sc","BBA","B.Com","BA","B.Des","BFA","MCA","MBA","M.Tech","M.Sc"])
        cgpa=st.number_input("CGPA",0.0,10.0,8.0,.1)
        skills=st.text_area("Skills","Python, SQL, Machine Learning")
        interests=st.text_area("Interests","AI, Technology, Data")
    with b:
        subjects=st.text_area("Favorite Subjects","Mathematics, Computer Science")
        personality=st.text_area("Personality","Analytical, Logical")
        goal=st.text_area("Career Goal","Technology, AI")
    go=st.form_submit_button("🔍 Find My Career",use_container_width=True)
if go:
    student={"Education":education,"Skills":skills,"Interests":interests,"Favorite Subjects":subjects,"Personality":personality,"Career Goal":goal}
    text=" ".join([skills,interests,subjects,personality,goal])
    p=model.predict_proba(vec.transform([text]))[0]
    rows=[]
    for c,prob in zip(model.classes_,p):
        rs=rule_score(student,c); rows.append([c,prob*100,rs,.75*prob*100+.25*rs])
    r=pd.DataFrame(rows,columns=["Career","ML Probability","Rule Score","Compatibility"]).sort_values("Compatibility",ascending=False).reset_index(drop=True)
    st.header("🏆 Top 3 Career Recommendations")
    for n,row in r.head(3).iterrows():
        st.markdown(f"## {n+1}. {row.Career}")
        st.progress(min(int(row.Compatibility),100)); st.write(f"**Compatibility: {row.Compatibility:.2f}%**")
        st.markdown("**Why recommended?**")
        for x in reasons(student,row.Career): st.write("✓",x)
        i=info(row.Career)
        if i: st.write("**Required Skills:**",i["Required Skills"]); st.write("**Roadmap:**",i["Roadmap"])
        st.markdown("---")
    st.subheader("📊 Career Ranking"); st.bar_chart(r.head(10).set_index("Career")[["Compatibility"]])
