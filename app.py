import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Iris AI - Premium", page_icon="🌺", layout="wide")

# ULTRA PREMIUM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');
html, body, [class*="css"]{font-family:'Poppins', sans-serif;}
.stApp{
  background: radial-gradient(circle at 20% 30%, #1a2980, #26d0ce);
  background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}
.glass{
  background: rgba(255,255,255,0.08);
  backdrop-filter: blur(15px);
  border-radius: 20px;
  padding: 25px;
  border: 1px solid rgba(255,255,255,0.15);
  box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
  color: white;
  transition: 0.3s;
}
.glass:hover{transform: translateY(-5px); box-shadow: 0 15px 40px rgba(0,0,0,0.5);}
.big-title{
  font-size: 50px!important; font-weight: 800;
  background: linear-gradient(90deg, #f6d365 0%, #fda085 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  text-align: center; letter-spacing: 1px;
}
.subtitle{text-align:center;color:#e0e0e0;font-size:18px;opacity:0.8}
.predict-box{
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 30px; border-radius: 25px; text-align: center;
  box-shadow: 0 0 30px rgba(102,126,234,0.6);
  animation: glow 2s infinite alternate;
}
@keyframes glow{from{box-shadow:0 0 20px rgba(102,126,234,0.6)} to{box-shadow:0 0 40px rgba(118,75,162,0.9)}}
.stSlider > div > div > div > div{background: linear-gradient(90deg, #f6d365, #fda085)!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🌺 IRIS FLOWER INTELLIGENCE 🌺</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-Gen AI that Predicts Flower Species with 93.33% Accuracy | Powered by Random Forest</div>', unsafe_allow_html=True)
st.write("")
st.write("")

# Load
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

# TOP 4 GLASS CARDS
c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(f'<div class="glass"><center><div style="font-size:35px">🎯</div><div style="font-size:13px;opacity:0.7">ACCURACY</div><div style="font-size:32px;font-weight:700;color:#f6d365">{acc*100:.1f}%</div></center></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="glass"><center><div style="font-size:35px">🌸</div><div style="font-size:13px;opacity:0.7">FLOWER TYPES</div><div style="font-size:32px;font-weight:700;color:#fda085">3 Species</div></center></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="glass"><center><div style="font-size:35px">📊</div><div style="font-size:13px;opacity:0.7">DATASET</div><div style="font-size:32px;font-weight:700;color:#a8edea">150 Samples</div></center></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="glass"><center><div style="font-size:35px">🤖</div><div style="font-size:13px;opacity:0.7">ALGORITHM</div><div style="font-size:20px;font-weight:700;color:#fed6e3;margin-top:10px">Random Forest<br>100 Trees</div></center></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.2,1], gap="large")

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('### 📈 Dataset Visualization')
    df_full = pd.DataFrame(iris.data, columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
    df_full['Species'] = [iris.target_names[i] for i in iris.target]
    fig = px.scatter_3d(df_full, x='Sepal Length', y='Sepal Width', z='Petal Length', color='Species',
                        color_discrete_sequence=['#f6d365','#fda085','#a8edea'], height=380)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', scene=dict(bgcolor='rgba(0,0,0,0)', xaxis=dict(color='white'), yaxis=dict(color='white'), zaxis=dict(color='white')), font=dict(color='white'), margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_full.head(8), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('### 🔮 AI Prediction Studio')
    st.markdown('<p style="opacity:0.7;font-size:13px">Adjust the sliders - AI will predict instantly</p>', unsafe_allow_html=True)

    sl = st.slider("🌿 Sepal Length", 4.0, 8.0, 5.1, 0.1)
    sw = st.slider("🍃 Sepal Width", 2.0, 4.5, 3.5, 0.1)
    pl = st.slider("🌸 Petal Length", 1.0, 7.0, 1.4, 0.1)
    pw = st.slider("💮 Petal Width", 0.1, 2.5, 0.2, 0.1)

    pred = model.predict([[sl, sw, pl, pw]])[0]
    proba = model.predict_proba([[sl, sw, pl, pw]])[0]
    species = iris.target_names[pred]

    colors = {"setosa": ["#FF6B6B","#FFE66D","🌼"], "versicolor": ["#4ECDC4","#44A08D","🌸"], "virginica": ["#667eea","#764ba2","🌺"]}
    col = colors[species]

    st.markdown(f"""
    <div class="predict-box" style="background: linear-gradient(135deg, {col[0]} 0%, {col[1]} 100%);">
        <div style="font-size:50px">{col[2]}</div>
        <div style="font-size:13px;letter-spacing:2px;opacity:0.8">PREDICTED SPECIES</div>
        <div style="font-size:34px;font-weight:800;color:white;margin:5px 0">{species.upper()}</div>
        <div style="font-size:14px;background:rgba(0,0,0,0.2);border-radius:20px;padding:5px 15px;display:inline-block">Confidence {max(proba)*100:.1f}% ✨</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    prob_df = pd.DataFrame({"Species": iris.target_names, "Probability": proba*100})
    fig2 = px.bar(prob_df, x="Species", y="Probability", color="Species", color_discrete_sequence=['#f6d365','#fda085','#a8edea'], height=200)
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), showlegend=False, margin=dict(l=0,r=0,t=0,b=0), yaxis=dict(color='white'))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.markdown("<center><p style='color:rgba(255,255,255,0.5);font-size:12px'>Crafted with 💖 by Pranathi | CodeAlpha Internship 2026 | Task-3 Machine Learning</p></center>", unsafe_allow_html=True)