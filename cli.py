import argparse
import sys
import soundfile as sf
from kanade_tokenizer import load_audio
from core.cloner import KokoClone
from core.chunked_convert import chunked_voice_conversion


def main() -> None:
    """Entry point for the KokoClone command-line tool."""
    parser = argparse.ArgumentParser(description="KokoClone: Zero-Shot Multilingual Voice Cloning")
    parser.add_argument(
        "--mode",
        choices=["tts", "convert"],
        default="tts",
        help="Operation mode: 'tts' (text → cloned speech) or 'convert' (audio → re-voiced speech)",
    )
    parser.add_argument("--text", type=str, help="[tts mode] Text to synthesize")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="[tts mode] Language code (en, hi, fr, ja, zh, it, pt, es)",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="[convert mode] Path to source audio file to re-voice (.wav)",
    )
    parser.add_argument("--ref", type=str, required=True, help="Path to reference voice audio file (.wav)")
    parser.add_argument("--out", type=str, default="output.wav", help="Output file path (.wav)")

    args = parser.parse_args()

    cloner = KokoClone()

    if args.mode == "tts":
        if not args.text:
            parser.error("--text is required when --mode is 'tts'")
        cloner.generate(
            text=args.text,
            lang=args.lang,
            reference_audio=args.ref,
            output_path=args.out,
        )

    elif args.mode == "convert":
        if not args.source:
            parser.error("--source is required when --mode is 'convert'")

        try:
            source_wav = load_audio(args.source, sample_rate=cloner.sample_rate).to(cloner.device)
            ref_wav = load_audio(args.ref, sample_rate=cloner.sample_rate).to(cloner.device)

            converted = chunked_voice_conversion(
                kanade=cloner.kanade,
                vocoder_model=cloner.vocoder,
                source_wav=source_wav,
                ref_wav=ref_wav,
                sample_rate=cloner.sample_rate,
            )

            sf.write(args.out, converted.numpy(), cloner.sample_rate)
            print(f"Success! Saved: {args.out}")
        except Exception as exc:
            print(f"Error during voice conversion: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()