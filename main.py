import cv2
import numpy as np

def main():
    # Load the built-in Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Age and mood estimation lists
    age_groups = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior', 'Elderly' , 'Very Elderly', 'Centenarian'
                  'Supercentenarian', 'Hypercentenarian','sexy', 'Nonagenarian']
    mood_list = ['Happy', 'Neutral', 'Sad', 'Surprised', 'Angry']
    

    # Initialize video capture with higher resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,550)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)  # Set height
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Face Detection with Age/Mood Estimation - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with adjusted parameters for close-up detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Reduced from 1.1 for better close-up detection
            minNeighbors=5,
            minSize=(100, 100),  # Increased minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Display face count
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Expand the detected rectangle by 20% to capture full face
            expansion = 0.2
            new_x = max(0, int(x - w * expansion/2))
            new_y = max(0, int(y - h * expansion/2))
            new_w = min(frame.shape[1]-new_x, int(w * (1 + expansion)))
            new_h = min(frame.shape[0]-new_y, int(h * (1 + expansion)))
            
            # Draw rectangle around face
            cv2.rectangle(frame, (new_x, new_y), (new_x+new_w, new_y+new_h), (0,255,0), 2)
            
            # Adjust age estimation for close-up faces
            age_index = min(int(new_w/80), len(age_groups)-1)  # Adjusted divisor
            age = age_groups[age_index]
            
            # Mood estimation
            mood_index = (new_x + new_y) % len(mood_list)
            mood = mood_list[mood_index]
            
             # Display age and mood above the expanded rectangle
            cv2.putText(frame, f"Age: {age}", (new_x, new_y -50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Mood: {mood}", (new_x, new_y -20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the output
        cv2.imshow('Face Detection', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()