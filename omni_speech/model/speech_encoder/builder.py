from .speech_encoder import WhisperWrappedEncoder, HubertEncoder


def build_speech_encoder(config):
    speech_encoder_type = getattr(config, 'speech_encoder_type')
    # if "whisper" in speech_encoder_type.lower():
    #     return WhisperWrappedEncoder.load(config)

    if speech_encoder_type == "hubert":
        return HubertEncoder.load(config)

    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')
