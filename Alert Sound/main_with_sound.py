import cv2
from ultralytics import YOLO
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

def estimate_distance(bbox_height):
    return 1000 / bbox_height

# الإعدادات
DISTANCE_THRESHOLD = 10.0  # المسافة الغير آمنة
ALERT_SOUND = 'sound.wav'  # ملف الصوت
video_path = 'somthing_wrong.mp4'  # مسار الفيديو
output_path = 'c100_out.mp4'  # مسار الفيديو الناتج

# فتح الفيديو
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# تحميل نموذج YOLO
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# إعداد فيديو الإخراج
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
if not out.isOpened():
    print("Error: Could not open VideoWriter.")
    exit()

# البدء في معالجة الفيديو
frame_number = 0
unsafe_frames = []
sound_played = False  # Variable to track if the sound has been played

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break
    frame_number += 1
    print(f"Processing frame {frame_number}")

    # استخدم YOLO لاكتشاف الكائنات
    results = model(frame)

    detected_unsafe_distance = False
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h = y2 - y1  # حساب ارتفاع الصندوق
            # التحقق إذا كانت السيارة أو المركبة مكتشفة بناءً على الفئات
            if int(box.cls[0]) in [2, 3, 5, 7] and box.conf[0] > 0.5:
                distance = estimate_distance(h)
                label = f"{distance:.2f}m"
                print(f"Detected vehicle at distance: {distance:.2f}m")  # Debug print

                # التحقق من المسافة الغير آمنة
                if distance < DISTANCE_THRESHOLD:
                    detected_unsafe_distance = True
                    print("Unsafe distance detected.")  # Debug print
                    # رسم الصندوق والنص باللون الأحمر
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    # رسم الصندوق والنص باللون الأخضر
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if detected_unsafe_distance:
        print("Unsafe distance detected in this frame.")  # Debug print
        unsafe_frames.append(frame_number)
        if not sound_played:
            sound_played = True
    else:
        print("No unsafe distance detected in this frame.")  # Debug print

    # كتابة الإطار في الفيديو الناتج
    out.write(frame)

# إغلاق كل شيء
cap.release()
out.release()

# إضافة الصوت إلى الفيديو الناتج باستخدام moviepy إذا تم اكتشاف مسافة غير آمنة
video_clip = VideoFileClip(output_path)
if unsafe_frames:
    audio_clip = AudioFileClip(ALERT_SOUND)
    final_clips = []
    sound_added = False  # Variable to track if the sound has been added
    for i, frame in enumerate(video_clip.iter_frames()):
        if i in unsafe_frames and not sound_added:
            frame_clip = video_clip.subclip(i / video_clip.fps, (i + 1) / video_clip.fps)
            frame_clip = frame_clip.set_audio(audio_clip)
            final_clips.append(frame_clip)
            sound_added = True  # Ensure the sound is added only once
        else:
            final_clips.append(video_clip.subclip(i / video_clip.fps, (i + 1) / video_clip.fps))
    final_clip = concatenate_videoclips(final_clips)
else:
    final_clip = video_clip

final_output_path = 'final_output_with_sound.mp4'
final_clip.write_videofile(final_output_path, codec='libx264', audio_codec='aac')

print("Processing complete. Output saved to:", final_output_path)