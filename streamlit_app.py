# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "17snWYz8vhgVomLL23UgzlHtE13ZgvIRT")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"]
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },
    labels[0] : {"text" : ["중국식 냉면은 맛있어"], "imgaes : ['https://www.google.com/imgres?q=%EC%A4%91%EA%B5%AD%EC%8B%9D%20%EB%83%89%EB%A9%B4&imgurl=https%3A%2F%2Fwww.kfoodtimes.com%2Fnews%2Fphoto%2F202106%2F16159_27527_3716.png&imgrefurl=https%3A%2F%2Fwww.kfoodtimes.com%2Fnews%2FarticleView.html%3Fidxno%3D16159&docid=_n2GOOMPNo2sxM&tbnid=qoZFfV4jYQrR0M&vet=12ahUKEwip5Pro7YmRAxVN1TQHHecxL4IQM3oECBYQAA..i&w=600&h=414&hcb=2&ved=2ahUKEwip5Pro7YmRAxVN1TQHHecxL4IQM3oECBYQAA']},
    labels[1] : {"text" : ["짜장면은 맛있어"], "imgaes : ['https://www.google.com/imgres?q=%EC%9E%90%EC%9E%A5%EB%A9%B4&imgurl=https%3A%2F%2Fimg.etoday.co.kr%2Fpto_db%2F2019%2F10%2F600%2F20191008103447_1374236_1200_800.jpg&imgrefurl=https%3A%2F%2Fbravo.etoday.co.kr%2Fview%2Fatc_view%2F9966&docid=LjngsFX_EOCKlM&tbnid=2rNEDwCow0pcXM&vet=12ahUKEwjhuLTS6YmRAxX3m68BHRNKETgQM3oECBYQAA..i&w=600&h=400&hcb=2&ved=2ahUKEwjhuLTS6YmRAxX3m68BHRNKETgQM3oECBYQAA']},
    labels[2] : {"text" : ["짬뽕은 맛있어"], "imgaes : ['https://www.google.com/imgres?q=%EC%A7%AC%EB%BD%95&imgurl=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FYPxRW%2FbtrzhpNljHH%2FAAAAAAAAAAAAAAAAAAAAAAhVpctCZeeRfUJSzJ9VBLKsQHsA38Gk5_KTV934P7vk%2Fimg.jpg%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1764514799%26allow_ip%3D%26allow_referer%3D%26signature%3DKtGzPuSD0MLN59%252BpAsKcHnaNZ0U%253D&imgrefurl=https%3A%2F%2Fgeniusjw.com%2F2610&docid=zCb-U9b0wq4nLM&tbnid=wFKu3Jk1dgeAvM&vet=12ahUKEwi-nuOY7omRAxWklFYBHYH1E_IQM3oECCsQAA..i&w=3315&h=2906&hcb=2&ved=2ahUKEwi-nuOY7omRAxWklFYBHYH1E_IQM3oECCsQAA']},
    labels[3] : {"text" : ["탕수육은 맛있어"], "imgaes : ['https://www.google.com/imgres?q=%ED%83%95%EC%88%98%EC%9C%A1&imgurl=https%3A%2F%2Fhomecuisine.co.kr%2Ffiles%2Fattach%2Fimages%2F142%2F737%2F002%2F969e9f7dc60d42510c5c0353a58ba701.JPG&imgrefurl=https%3A%2F%2Fhomecuisine.co.kr%2Fhc20%2F2737%2Fpage%2F13%3Fis_keyword%3D%25EA%25B0%2584%25EC%25A7%259C%25EC%259E%25A5%26where%3Dcomment%26search_target%3Dcontent%26m%3D0%26listStyle%3Dwebzine%26sort_index%3Dreaded_count%26order_type%3Ddesc&docid=6-oydAf7Yom97M&tbnid=yHaN8xvTC09UGM&vet=12ahUKEwiT-sCk7omRAxXj8DQHHeXVJJYQM3oECBEQAA..i&w=930&h=618&hcb=2&itg=1&ved=2ahUKEwiT-sCk7omRAxXj8DQHHeXVJJYQM3oECBEQAA']},
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
