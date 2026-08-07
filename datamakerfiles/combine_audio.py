import os
from pydub import AudioSegment

# PriMock57 dataset audio -- doctor/patient tracks recorded as separate mono
# files per consultation, combined here into one stereo file per
# consultation (Left = Doctor, Right = Patient).
audio_dir = "../data/primock57/audio"
output_dir = "../data/primock57/audio_combined"
os.makedirs(output_dir, exist_ok=True)


def _match_sample_count(a, b):
    """Doctor/patient tracks are recorded on separate channels and can
    differ by a handful of samples (confirmed directly: consultation01's
    pair differs by 960 samples, ~60ms) -- from_mono_audiosegments requires
    an EXACT sample-count match. Truncates both to the shorter one's exact
    sample count (not a millisecond-rounded duration, which wouldn't
    guarantee exact parity) rather than padding, since losing <100ms off
    one end of a multi-minute consultation recording is negligible."""
    samples_a = a.get_array_of_samples()
    samples_b = b.get_array_of_samples()
    n = min(len(samples_a), len(samples_b))
    return a._spawn(samples_a[:n]), b._spawn(samples_b[:n])


# Loop through all doctor audio files
for file in os.listdir(audio_dir):
    if file.endswith("_doctor.wav"):
        base_name = file.replace("_doctor.wav", "")
        doctor_path = os.path.join(audio_dir, file)
        patient_path = os.path.join(audio_dir, f"{base_name}_patient.wav")

        if os.path.exists(patient_path):
            doctor_audio = AudioSegment.from_wav(doctor_path)
            patient_audio = AudioSegment.from_wav(patient_path)

            if len(doctor_audio.get_array_of_samples()) != len(patient_audio.get_array_of_samples()):
                doctor_audio, patient_audio = _match_sample_count(doctor_audio, patient_audio)

            # Combine into stereo: Left = Doctor, Right = Patient
            stereo_audio = AudioSegment.from_mono_audiosegments(doctor_audio, patient_audio)

            output_path = os.path.join(output_dir, f"{base_name}_stereo.wav")
            stereo_audio.export(output_path, format="wav")
            print(f"Combined: {base_name}_stereo.wav")
