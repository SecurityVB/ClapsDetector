from GeneratorDataset.AudioManager import *
from config import *

def create_aug_files():
    # for file in os.listdir(path_claps_m4a):
    #     if file.endswith("m4a"):
    #         audio = AudioSegment.from_file(path_claps_m4a+file, format="m4a")
    #         file = file.split('.')[0]
    #         save(f'{file}.wav', audio, path_claps_wav)
    #         for db in range(-15, 16, 5):
    #             aug_audio = change_volume(audio, db)
    #             save(f'{file}_db_{db}.wav', aug_audio, path_claps_wav)
    #         for i in (0.01, 0.02, 0.03):
    #             aug_audio = add_noise(audio, i)
    #             save(f'{file}_noise_{i}.wav', aug_audio, path_claps_wav)
    #         for db in range(-15, 16, 5):
    #             if db != 0:
    #                 for i in (0.01, 0.02, 0.03):
    #                     aug_audio = change_volume(add_noise(audio, i), db)
    #                     save(f'{file}_noise_{i}_db_{db}.wav', aug_audio, path_claps_wav)

    """=================NO_CLAPS==============="""

    for file in os.listdir(path_Noclaps_m4a):
        if file.endswith("m4a") and ("Noise" in file) and ("_" not in file):
            audio = AudioSegment.from_file(path_Noclaps_m4a+file, format="m4a")
            file = file.split('.')[0]
            save(f'{file}.wav', audio, path_Noclaps_wav)
            print(f"Create {file}.wav")
            for db in range(-15, 16, 5):
                aug_audio = change_volume(audio, db)
                save(f'{file}_db_{db}.wav', aug_audio, path_Noclaps_wav)
            for i in (0.01, 0.02, 0.03):
                aug_audio = add_noise(audio, i)
                save(f'{file}_noise_{i}.wav', aug_audio, path_Noclaps_wav)
            for db in range(-15, 16, 5):
                if db != 0:
                    for i in (0.01, 0.02, 0.03):
                        aug_audio = change_volume(add_noise(audio, i), db)
                        save(f'{file}_noise_{i}_db_{db}.wav', aug_audio, path_Noclaps_wav)