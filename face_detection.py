from retinaface import RetinaFace
import cv2

def extract_faces(image_path, output_dir):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Perform face detection using RetinaFace
    faces = RetinaFace.detect_faces(image)

    if faces is not None:
        for i, face in enumerate(faces):
            x, y, x2, y2, score = face['facial_area']
            face_image = image[y:y2, x:x2]
            output_filename = f"{output_dir}/face_{i}.jpg"
            cv2.imwrite(output_filename, face_image)

if __name__ == "__main__":
    image_path = "Dest_mod.jpg"
    output_dir = "output_faces"
    extract_faces(image_path, output_dir)
