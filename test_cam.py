import cv2

# Try to open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera.")
else:
    print("✅ SUCCESS: Camera detected! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Camera Test', frame)
    
    # Press 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()