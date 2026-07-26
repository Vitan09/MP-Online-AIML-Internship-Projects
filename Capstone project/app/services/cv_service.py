import cv2


class CVService:
    """
    Computer Vision Service

    Features:
    1. Webcam Capture
    2. Image Resize
    3. Grayscale Conversion
    4. Gaussian Blur
    5. Canny Edge Detection
    6. Face Detection using Haar Cascade
    """

    def __init__(self):

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    # -------------------------
    # Resize Image
    # -------------------------
    def resize_image(self, image, width=640, height=480):
        return cv2.resize(image, (width, height))

    # -------------------------
    # Convert to Grayscale
    # -------------------------
    def convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # Gaussian Blur
    # -------------------------
    def apply_blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    # -------------------------
    # Edge Detection
    # -------------------------
    def detect_edges(self, image):
        return cv2.Canny(image, 100, 200)

    # -------------------------
    # Face Detection
    # -------------------------
    def detect_faces(self, image):

        gray = self.convert_to_grayscale(image)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return faces

    # -------------------------
    # Draw Face Boxes
    # -------------------------
    def draw_faces(self, image):

        faces = self.detect_faces(image)

        for (x, y, w, h) in faces:

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        return image, len(faces)

    # -------------------------
    # Webcam Demo
    # -------------------------
    def start_webcam(self):

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("Unable to open webcam.")
            return

        print("Press 'Q' to Quit")

        while True:

            success, frame = camera.read()

            if not success:
                break

            # Resize
            resized = self.resize_image(frame)

            # Gray
            gray = self.convert_to_grayscale(resized)

            # Blur
            blurred = self.apply_blur(gray)

            # Edge
            edges = self.detect_edges(blurred)

            # Face Detection
            face_image, total_faces = self.draw_faces(resized.copy())

            cv2.putText(
                face_image,
                f"Faces Detected : {total_faces}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow("Original", resized)
            cv2.imshow("Grayscale", gray)
            cv2.imshow("Blur", blurred)
            cv2.imshow("Canny Edge Detection", edges)
            cv2.imshow("Face Detection", face_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    cv = CVService()
    cv.start_webcam()