import threading
import time
import cv2
import gradio as gr
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("best.pt")


class VideoStreamHandler:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(source)

        self.ret, self.frame = False, None
        self.stopped = False
        self.lock = threading.Lock()

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or self.fps > 120:
            self.fps = 30
        self.frame_delay = 1.0 / self.fps

    def start(self):
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            with self.lock:
                self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

    def get_fps(self):
        return self.fps


def stream_detection(source, is_webcam=False):
    streamer = VideoStreamHandler(source).start()
    time.sleep(1.0)

    video_fps = streamer.get_fps()
    frame_time = 1.0 / video_fps
    last_frame_time = time.time()

    while not streamer.stopped:
        frame = streamer.read()
        if frame is None:
            continue

        results = model(frame, imgsz=320, verbose=False)
        annotated_frame = results[0].plot()

        # Count objects in the current frame
        boxes = results[0].boxes
        frame_counts = defaultdict(int)
        if boxes is not None and boxes.cls is not None:
            for cls_id in boxes.cls:
                class_name = model.names[int(cls_id)]
                frame_counts[class_name] += 1

        # Convert BGR → RGB for Gradio
        output_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        if not is_webcam:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_frame_time = time.time()

        # Format counts as Markdown
        md_stats = "### Current Frame Object Counts\n"
        if frame_counts:
            for k, v in frame_counts.items():
                md_stats += f"- **{k}**: {v}\n"
        else:
            md_stats += "_No objects detected_\n"

        yield output_frame, md_stats

    streamer.stop()


def stream_video(video_file):
    yield from stream_detection(video_file, is_webcam=False)


def stream_webcam():
    yield from stream_detection(0, is_webcam=True)


with gr.Blocks() as app:
    gr.Markdown("## 🎥 Real-Time YOLO Object Detection (Per-Frame Stats)")

    with gr.Tabs():
        with gr.Tab("📁 Video"):
            gr.Markdown("Upload a video file to start detection")
            video_output = gr.Image(streaming=True, label="Detection Output")
            video_stats = gr.Markdown(label="Detection Statistics")
            video_upload = gr.UploadButton("Upload Video", file_types=["video"], file_count="single")
            video_upload.upload(
                fn=stream_video,
                inputs=video_upload,
                outputs=[video_output, video_stats]
            )

        with gr.Tab("📷 Webcam"):
            gr.Markdown("Click to start real-time webcam detection")
            cam_output = gr.Image(streaming=True, label="Live Detection")
            cam_stats = gr.Markdown(label="Detection Statistics")
            start_cam = gr.Button("Start Webcam")
            start_cam.click(
                fn=stream_webcam,
                outputs=[cam_output, cam_stats]
            )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)