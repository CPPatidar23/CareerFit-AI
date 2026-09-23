import pandas as pd, re
def kb(): return pd.read_csv("career_knowledge.csv")
def info(career):
    x=kb(); r=x[x.Career==career]
    return None if r.empty else r.iloc[0].to_dict()
def reasons(student,career):
    i=info(career)
    if i is None:return []
    out=[]
    pairs=[("Skills","Required Skills","Your skills match important skills for this career."),
           ("Interests","Interests","Your interests align with this career."),
           ("Favorite Subjects","Favorite Subjects","Your favorite subjects support this career."),
           ("Personality","Personality","Your personality characteristics fit this career."),
           ("Education","Education","Your education background is suitable for this career.")]
    for sc,kc,msg in pairs:
        s=str(student.get(sc,"")).lower()
        terms=[x.strip().lower() for x in str(i[kc]).split(",")]
        if any(t and t in s for t in terms): out.append(msg)
    return out[:5]
def rule_score(student,career):
    i=info(career)
    if i is None:return 0
    pairs=[("Skills","Required Skills"),("Interests","Interests"),("Favorite Subjects","Favorite Subjects"),("Personality","Personality"),("Education","Education")]
    score=0
    for sc,kc in pairs:
        s=str(student.get(sc,"")).lower()
        terms=[x.strip().lower() for x in str(i[kc]).split(",")]
        score += min(sum(t in s for t in terms),3)/3*20
    return round(min(score,100),2)
