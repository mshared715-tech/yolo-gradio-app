import os
import cv2
import gradio as gr
from ultralytics import YOLO
from collections import defaultdict

# ================================
# Load YOLO model
# ================================
model = YOLO("best.pt")

custom_css = """
#cam_output label {
    color: black !important;
}

#cam_preview label {
    color: black !important;
}
"""
# ================================
# DETECTION FUNCTION
# ================================
def detect_and_count(frame, convert_to_rgb=False):

    resize_width = 640
    h, w = frame.shape[:2]
    scale = resize_width / w
    frame = cv2.resize(frame, (resize_width, int(h * scale)))

    results = model(frame, imgsz=224, verbose=False)

    annotated = results[0].plot()
    boxes = results[0].boxes

    counts = defaultdict(int)

    if boxes is not None and boxes.cls is not None:
        for cls_id in boxes.cls:
            class_name = model.names[int(cls_id)]
            counts[class_name] += 1

    # Fix color depending on source
    if convert_to_rgb:
        output_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    else:
        output_frame = annotated

    stats = "###  Live Detection Statistics\n"

    if counts:
        for k, v in counts.items():
            stats += f"- **{k}** : {v}\n"
    else:
        stats += "_No objects detected_\n"

    return output_frame, stats



# ================================
# WEBCAM STREAM
# ================================
def webcam_stream(frame):

    if frame is None:
        return None, "No frame received"

    # Webcam already correct color
    return detect_and_count(frame, convert_to_rgb=False)


# ================================
# VIDEO STREAM
# ================================
def video_stream(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        yield None, "Could not open video"
        return

    frame_skip = 4
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 360))

        # Convert color for video frames
        output_frame, stats = detect_and_count(frame, convert_to_rgb=True)

        yield output_frame, stats

    cap.release()


# ================================
# UI DASHBOARD
# ================================
with gr.Blocks(theme=gr.themes.Soft(),css=custom_css) as app:

    gr.Markdown("# Object Detection System")

    with gr.Tabs():

        # ================================
        # WEBCAM TAB
        # ================================
        with gr.Tab("📷 Live Webcam Detection"):

            with gr.Column():

                cam_output = gr.Image(
                    label="Detection Output",
                    height=550,
                     elem_id="cam_output"
                )

                with gr.Row():

                    with gr.Column(scale=1):
                        cam_input = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            label="Webcam Preview",
                            height=160,
                            elem_id="cam_preview"
                        )

                    with gr.Column(scale=1):
                        cam_stats = gr.Markdown(
                            "### Detection Statistics"
                        )

            cam_input.stream(
                fn=webcam_stream,
                inputs=cam_input,
                outputs=[cam_output, cam_stats]
            )
            


        # ================================
        # VIDEO TAB
        # ================================
        with gr.Tab("📁 Upload Video Detection"):

            with gr.Column():

                video_output = gr.Image(
                    label="Detection Output",
                    height=550
                )

                with gr.Row():

                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Upload Video"
                        )

                    with gr.Column(scale=1):
                        video_stats = gr.Markdown(
                            "### Live Detection Statistics"
                        )

            video_input.change(
                fn=video_stream,
                inputs=video_input,
                outputs=[video_output, video_stats]
            )


# ================================
# LAUNCH APP
# ================================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7861))

    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )