import cv2
import ultralytics
import torch
import logging
import os

from plot_boxes import plot_bboxes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(torch.cuda.is_available())
ultralytics.checks()
logging.info(cv2.__version__)


class YOLODetector:
    def __init__(self, yolov8_path: str = 'yolov8x.pt', conf: float = 0.5, filter_labels: set[str] | None = None,
                 labels=None, colors=None, output_dir='output'):
        self.config = {
            'conf': conf,
            'filter_labels': filter_labels,
            'labels': labels,
            'colors': colors
        }
        self.yolov8_path = yolov8_path
        self.output_dir = output_dir  # Separate output_dir from config to avoid passing it to plot_bboxes
        self.model = self.get_model()
        os.makedirs(output_dir, exist_ok=True)

    def get_model(self) -> ultralytics.YOLO:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return ultralytics.YOLO(self.yolov8_path)

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            self.config[key] = value

    def show_config(self):
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print(f"Output directory: {self.output_dir}")

    def run_video(self, video_path: str, save_output=False) -> None:
        cap = cv2.VideoCapture(video_path)
        self.process_stream(cap, save_output, os.path.basename(video_path))

    def run_webcam(self, save_output=False) -> None:
        cap = cv2.VideoCapture(0)
        self.process_stream(cap, save_output, "webcam_output.mp4")

    def run_image(self, image_path: str, save_output=False) -> None:
        frame = cv2.imread(image_path)
        if frame is None:
            logging.error(f"Não foi possível carregar a imagem: {image_path}")
            return
        self.process_frame(frame, save_output, os.path.basename(image_path))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_stream(self, cap, save_output, file_name):
        out = None
        if save_output:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            # Ensure the resolution is correctly captured
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = os.path.join(self.output_dir, file_name)
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height), True)

            # Check if VideoWriter was successfully initialized
            if not out.isOpened():
                logging.error(f"Failed to open video writer for the file: {out_path}")
                save_output = False  # Disable saving because the VideoWriter didn't initialize properly

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.process_frame(frame, save_output, file_name, out if save_output else None)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            if save_output and out is not None:
                out.release()
            cv2.destroyAllWindows()

    def process_frame(self, frame, save_output, file_name, out=None):
        results = self.model.predict(frame)
        plot_bboxes(frame, results[0].boxes.data, **self.config)
        cv2.imshow('Detecção de Objetos', frame)
        if save_output:
            if out:
                out.write(frame)
            else:
                cv2.imwrite(os.path.join(self.output_dir, file_name), frame)


if __name__ == '__main__':
    detector = YOLODetector()
    detector.show_config()
    detector.run_video('videos/carolsdogshort.mp4', save_output=True)
    # detector.run_image('images/gato.jpeg', save_output=True)
