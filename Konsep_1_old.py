import streamlit as st
import cv2
import numpy as np
import fitz  # PyMuPDF
import mediapipe as mp
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import io
from PIL import Image
from datetime import datetime
import os
import uuid

# ============================================
# 1. Konfigurasi Halaman & Variabel Sesi
# ============================================
st.set_page_config(page_title="Gaze Heatmap Analyzer", layout="wide")
st.title("👁️ PDF Gaze Heatmap Analyzer (Pro Version)")
st.write("Dilengkapi dengan Kalibrasi Mata, Nama Responden, dan Perekaman Video.")

if "final_npy_data" not in st.session_state:
    st.session_state.final_npy_data = None
if "final_video_bytes" not in st.session_state:
    st.session_state.final_video_bytes = None

# ============================================
# 2. Sidebar - Panel Kustomisasi & Data
# ============================================
with st.sidebar:
    st.header("📋 Data & Kalibrasi")
    
    # 1. Nama Responden
    responden_id = st.text_input("Nama / ID Responden", value="Responden_01")
    
    st.divider()
    st.write("⚙️ **Kalibrasi Sensitivitas Mata**")
    st.caption("Ubah nilai ini jika lirikan mata terasa kurang pas dengan ukuran layar.")
    
    col1, col2 = st.columns(2)
    with col1:
        min_x_val = st.slider("Batas Kiri", 0.1, 0.5, 0.35, step=0.01)
        min_y_val = st.slider("Batas Atas", 0.1, 0.5, 0.30, step=0.01)
    with col2:
        max_x_val = st.slider("Batas Kanan", 0.5, 0.9, 0.65, step=0.01)
        max_y_val = st.slider("Batas Bawah", 0.5, 0.9, 0.70, step=0.01)
        
    st.divider()
    st.write("🎨 **Ketebalan Heatmap**")
    heat_threshold = st.slider("Batas Intensitas Maksimal", 10, 100, 30)

# ============================================
# 3. Upload & Render PDF
# ============================================
uploaded_file = st.file_uploader("1. Upload Dokumen PDF (.pdf)", type=["pdf"])

pdf_frame = None
vid_height, vid_width = 0, 0

if uploaded_file is not None:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=150)
    
    pdf_frame = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        pdf_frame = cv2.cvtColor(pdf_frame, cv2.COLOR_RGBA2BGR)
    else:
        pdf_frame = cv2.cvtColor(pdf_frame, cv2.COLOR_RGB2BGR)
        
    vid_height, vid_width = pdf_frame.shape[:2]
    st.success(f"PDF Berhasil dimuat! Resolusi: {vid_width} x {vid_height}")

# ============================================
# 4. Kelas Pemroses Video WebRTC
# ============================================
class GazeVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.cumulative_heatmap = None
        self.pdf_bg = None
        self.w = 0
        self.h = 0
        
        # Variabel Kalibrasi
        self.min_x, self.max_x = 0.35, 0.65
        self.min_y, self.max_y = 0.30, 0.70
        
        # Variabel Perekaman Video
        self.video_writer = None
        self.video_path = f"temp_video_{uuid.uuid4().hex}.mp4"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.pdf_bg is None:
            return frame

        if self.cumulative_heatmap is None:
            self.cumulative_heatmap = np.zeros((self.h, self.w), dtype=np.float32)

        temp_gaze = np.zeros((self.h, self.w), dtype=np.float32)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                iris = lm.landmark[473]
                eye_outer = lm.landmark[263]
                eye_inner = lm.landmark[362]
                eye_top = lm.landmark[386]
                eye_bottom = lm.landmark[374]

                eye_width = max(abs(eye_inner.x - eye_outer.x), 0.001)
                eye_height = max(abs(eye_bottom.y - eye_top.y), 0.001)

                gaze_x_ratio = (iris.x - min(eye_outer.x, eye_inner.x)) / eye_width
                gaze_y_ratio = (iris.y - min(eye_top.y, eye_bottom.y)) / eye_height

                mapped_x = (gaze_x_ratio - self.min_x) / (self.max_x - self.min_x)
                mapped_y = (gaze_y_ratio - self.min_y) / (self.max_y - self.min_y)

                mapped_x = max(0.0, min(1.0, mapped_x))
                mapped_y = max(0.0, min(1.0, mapped_y))

                ix = int(mapped_x * self.w)
                iy = int(mapped_y * self.h)
                
                cv2.circle(temp_gaze, (ix, iy), 60, (1.0,), thickness=-1)
                self.cumulative_heatmap += temp_gaze

        # -- PEREKAMAN VIDEO (Di Latar Belakang) --
        # 1. Inisialisasi Writer Jika Belum Ada
        if self.video_writer is None and self.w > 0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 15.0, (self.w, self.h))
        
        # 2. Render Efek Kumulatif untuk Frame Video
        capped_for_vid = np.clip(self.cumulative_heatmap, 0, 30)
        cum_blurred = cv2.GaussianBlur(capped_for_vid, (0, 0), sigmaX=50, sigmaY=50)
        if np.max(cum_blurred) > 0:
            cum_norm = cv2.normalize(cum_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cum_color = cv2.applyColorMap(cum_norm, cv2.COLORMAP_JET)
            alpha_cum = cum_norm.astype(np.float32) / 255.0 * 0.6 
            alpha_cum = np.expand_dims(alpha_cum, axis=-1)
            frame_to_record = self.pdf_bg.astype(np.float32) * (1 - alpha_cum) + cum_color.astype(np.float32) * alpha_cum
            frame_to_record = frame_to_record.astype(np.uint8)
        else:
            frame_to_record = self.pdf_bg.copy()
            
        # 3. Tulis Frame ke Video File
        if self.video_writer is not None:
            self.video_writer.write(frame_to_record)

        # -- TAMPILAN UI TERSEMBUNYI --
        ui_display = self.pdf_bg.copy()
        cv2.circle(ui_display, (40, 40), 15, (0, 0, 255), -1) 
        
        ui_display_resized = cv2.resize(ui_display, (img.shape[1], img.shape[0]))
        import av
        return av.VideoFrame.from_ndarray(ui_display_resized, format="bgr24")

# ============================================
# 5. Modul Kamera WebRTC
# ============================================
if pdf_frame is not None:
    st.write("### 2. Live Tracking")
    st.info("Nyalakan kamera, tunggu hingga PDF muncul. Saat selesai merekam, matikan kamera untuk menghasilkan tombol Download.")
    
    webrtc_ctx = webrtc_streamer(
        key="gaze-tracker",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=GazeVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Hubungkan Slider Sidebar ke Processor
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.pdf_bg = pdf_frame
        webrtc_ctx.video_processor.w = vid_width
        webrtc_ctx.video_processor.h = vid_height
        webrtc_ctx.video_processor.min_x = min_x_val
        webrtc_ctx.video_processor.max_x = max_x_val
        webrtc_ctx.video_processor.min_y = min_y_val
        webrtc_ctx.video_processor.max_y = max_y_val

    # Saat Kamera Dimatikan, Ekstrak Data
    if not webrtc_ctx.state.playing and webrtc_ctx.video_processor and webrtc_ctx.video_processor.cumulative_heatmap is not None:
        
        # 1. Matikan dan Ekstrak Video
        if webrtc_ctx.video_processor.video_writer:
            webrtc_ctx.video_processor.video_writer.release()
            
            # Baca file MP4 sementara ke memori
            if os.path.exists(webrtc_ctx.video_processor.video_path):
                with open(webrtc_ctx.video_processor.video_path, 'rb') as v_file:
                    st.session_state.final_video_bytes = v_file.read()
                # Bersihkan file sampah
                os.remove(webrtc_ctx.video_processor.video_path)
        
        # 2. Simpan Data NPY (Mentah)
        st.session_state.final_npy_data = webrtc_ctx.video_processor.cumulative_heatmap
        st.success("Data berhasil diproses! Scroll ke bawah untuk melihat hasil.")

# ============================================
# 6. Area Download & Live Preview
# ============================================
if st.session_state.final_npy_data is not None:
    st.write("---")
    st.write(f"### 3. Hasil Data: {responden_id}")
    
    # Render Gambar Heatmap secara Dinamis berdasarkan SLIDER Ketebalan
    capped_heatmap = np.clip(st.session_state.final_npy_data, 0, heat_threshold)
    cum_blurred = cv2.GaussianBlur(capped_heatmap, (0, 0), sigmaX=50, sigmaY=50)
    
    if np.max(cum_blurred) > 0:
        cum_norm = cv2.normalize(cum_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cum_color = cv2.applyColorMap(cum_norm, cv2.COLORMAP_JET)
        alpha_cum = cum_norm.astype(np.float32) / 255.0 * 0.6 
        alpha_cum = np.expand_dims(alpha_cum, axis=-1)
        
        final_cum_img = pdf_frame.astype(np.float32) * (1 - alpha_cum) + cum_color.astype(np.float32) * alpha_cum
        final_cum_img = final_cum_img.astype(np.uint8)
    else:
        final_cum_img = pdf_frame.copy()

    # Tampilkan Preview
    preview_rgb = cv2.cvtColor(final_cum_img, cv2.COLOR_BGR2RGB)
    st.image(preview_rgb, width=500, caption=f"Preview Heatmap - Intensitas: {heat_threshold}")
    
    # Persiapkan Byte Data untuk Tombol Download
    img_pil = Image.fromarray(preview_rgb)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    
    npy_buf = io.BytesIO()
    np.save(npy_buf, st.session_state.final_npy_data)
    npy_bytes = npy_buf.getvalue()
    
    # Penamaan File Otomatis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{responden_id.replace(' ', '_')}_{timestamp}"
    
    # Baris Tombol Download
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="🖼️ Download JPG",
            data=byte_im,
            file_name=f"heatmap_{base_name}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="💾 Download NPY",
            data=npy_bytes,
            file_name=f"data_{base_name}.npy",
            mime="application/octet-stream",
            use_container_width=True
        )
    with col3:
        if st.session_state.final_video_bytes:
            st.download_button(
                label="🎥 Download Video MP4",
                data=st.session_state.final_video_bytes,
                file_name=f"video_{base_name}.mp4",
                mime="video/mp4",
                use_container_width=True
            )