from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
            BmfCallBackType, VideoFrame, AudioFrame
from bmf.lib._bmf.sdk import ffmpeg
from bmf.hml import hmp as mp

from dlib_alignment import dlib_detect_face, face_recover
import torchvision.transforms as transforms
from models.SRGAN_model import SRGANModel
import argparse
import numpy as np
import torch
from PIL import Image


_transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])])


class FaceSROpt:
    gpu_ids = None
    batch_size = 32
    lr_G = 1e-4
    weight_decay_G = 0
    beta1_G = 0.9
    beta2_G = 0.99
    lr_D = 1e-4
    weight_decay_D = 0
    beta1_D = 0.9
    beta2_D = 0.99
    lr_scheme = 'MultiStepLR'
    niter = 100000
    warmup_iter = -1
    lr_steps = [50000]
    lr_gamma = 0.5
    pixel_criterion = 'l1'
    pixel_weight = 1e-2
    feature_criterion = 'l1'
    feature_weight = 1
    gan_type = 'ragan'
    gan_weight = 5e-3
    D_update_ratio = 1
    D_init_iters = 0
    print_freq = 100
    val_freq = 1000
    save_freq = 10000
    crop_size = 0.85
    lr_size = 128
    hr_size = 512
    which_model_G = 'RRDBNet'
    G_in_nc = 3
    out_nc = 3
    G_nf = 64
    nb = 16
    which_model_D = 'discriminator_vgg_128'
    D_in_nc = 3
    D_nf = 64
    pretrain_model_G = '90000_G.pth'
    pretrain_model_D = None


class FaceSR(Module):
    def __init__(self, node, option=None):
        self.sr_model = SRGANModel(FaceSROpt(), is_train=False)
        self.sr_model.load()

    def process(self, task):
        input_packets = task.get_inputs()[0]
        output_packets = task.get_outputs()[0]

        while not input_packets.empty():
            pkt = input_packets.get()

            if pkt.timestamp == Timestamp.EOF:
                Log.log_node(LogLevel.DEBUG, task.get_node(), "Receive EOF")
                output_packets.put(Packet.generate_eof_packet())
                task.timestamp = Timestamp.DONE
                return ProcessResult.OK

            if pkt.is_(VideoFrame) and pkt.timestamp != Timestamp.UNSET:
                vf = pkt.get(VideoFrame)
                frame = ffmpeg.reformat(vf, "rgb24").frame().plane(0).numpy()

                sr_frame = self.sr_forward(frame)

                rgb = mp.PixelInfo(mp.kPF_RGB24)
                video_frame = VideoFrame(mp.Frame(mp.from_numpy(sr_frame), rgb))
                video_frame.pts = vf.pts
                video_frame.time_base = vf.time_base
                out_pkt = Packet(video_frame)
                out_pkt.timestamp = video_frame.pts
                output_packets.put(out_pkt)

        return ProcessResult.OK

    def sr_forward(self, img, padding=0.5, moving=0.1):
        img_aligned, M = dlib_detect_face(img, padding=padding, image_size=(128, 128), moving=moving)
        input_img = torch.unsqueeze(_transform(Image.fromarray(img_aligned)), 0)
        self.sr_model.var_L = input_img.to(self.sr_model.device)
        self.sr_model.test()
        output_img = self.sr_model.fake_H.squeeze(0).cpu().numpy()
        output_img = np.clip((np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        rec_img = face_recover(output_img, M * 4, img)
        return rec_img

