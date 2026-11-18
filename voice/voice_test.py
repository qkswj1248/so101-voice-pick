
import speech_recognition as sr
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 4

COLORS = {
    "red": ["빨간", "빨강", "레드"],
    "blue": ["파란", "파랑", "블루"],
    "yellow": ["노란", "노랑", "옐로"],
    "green": ["초록", "녹색", "그린"],
    "orange": ["주황", "오렌지"],
    "purple": ["보라", "퍼플"],
}

recognizer = sr.Recognizer()


def extract_color(text):
    for color, keywords in COLORS.items():
        if any(keyword in text for keyword in keywords):
            return color

    return None


def recognize_voice():
    print("\n🎤 명령을 말씀하세요 (4초 녹음)")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    audio_data = sr.AudioData(
        audio.tobytes(),
        SAMPLE_RATE,
        2
    )

    try:
        text = recognizer.recognize_google(
            audio_data,
            language="ko-KR"
        )

        print(f"인식된 음성: {text}")

        color = extract_color(text)

        if color is None:
            print("색상을 찾지 못했습니다.")
        else:
            print(f"선택된 색상: {color}")
            print(f"target_color = '{color}'")

        return color

    except sr.UnknownValueError:
        print("음성을 인식하지 못했습니다.")

    except sr.RequestError as e:
        print(f"음성 인식 서비스 오류: {e}")

    return None


if __name__ == "__main__":
    print("=== SO-ARM Voice Recognition ===")

    while True:
        command = input(
            "\nEnter = 음성 명령 / q = 종료: "
        ).strip().lower()

        if command == "q":
            break

        recognize_voice()
