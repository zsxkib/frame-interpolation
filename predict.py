import os
import glob
import shutil
import subprocess
import numpy as np
import tensorflow as tf
from cog import BasePredictor, Input
from cog import Path as CogPath
from eval import interpolator as film_interpolator, util as film_util


class Predictor(BasePredictor):
    def setup(self):
        print("Loading interpolator...")
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.interpolator = film_interpolator.Interpolator(
            # from https://drive.google.com/drive/folders/1i9Go1YI2qiFWeT5QtywNFmYAA74bhXWj?usp=sharing
            "/src/frame_interpolation_saved_model",
            None,
        )

    def predict(
        self,
        mp4: CogPath = Input(
            description="Provide an mp4 video file for frame interpolation.",
        ),
        playback_frames_per_second: int = Input(
            description="Specify the playback speed in frames per second.",
            default=24,
            ge=1,
            le=60,
        ),
        num_interpolation_steps: int = Input(
            description="Number of steps to interpolate between animation frames",
            default=3,
            ge=1,
            le=50,
        ),
    ) -> CogPath:
        path_to_mp4 = str(mp4)
        output_dir = "frames"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        output_pattern = os.path.join(output_dir, "%04d.png")

        try:
            subprocess.run(
                ["ffmpeg", "-i", path_to_mp4, output_pattern],
                check=True,
            )
        except Exception as e:
            print(f"Error running ffmpeg: {e}")
            return None

        original_frame_filenames = sorted(glob.glob(os.path.join(output_dir, "*.png")))

        print("Interpolating frames with FILM...")
        interpolated_frames = film_util.interpolate_recursively_from_files(
            original_frame_filenames, num_interpolation_steps, self.interpolator
        )
        interpolated_frames = list(interpolated_frames)

        interpolated_frames_dir = os.path.join(output_dir, "interpolated_frames")
        if os.path.exists(interpolated_frames_dir):
            shutil.rmtree(interpolated_frames_dir)
        os.makedirs(interpolated_frames_dir)

        for i, frame in enumerate(interpolated_frames):
            frame_filename = os.path.join(interpolated_frames_dir, f"{i:08d}.png")
            film_util.write_image(str(frame_filename), frame)

        input_pattern = os.path.join(interpolated_frames_dir, "%08d.png")
        output_video = os.path.join(output_dir, "output_video.mp4")
        ffmpeg_command = [
            "ffmpeg",
            "-r",
            str(playback_frames_per_second),
            "-i",
            input_pattern,
            "-vcodec",
            "libx264",
            "-crf",
            "1",
            "-pix_fmt",
            "yuv420p",
            output_video,
        ]

        try:
            subprocess.run(ffmpeg_command, check=True)
        except Exception as e:
            print(f"Error running ffmpeg: {e}")
            return None

        return CogPath(output_video)
