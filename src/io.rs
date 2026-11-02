//! Audio input and output, entirely in-process through pure-Rust codecs.
//!
//! No external `ffmpeg`, no subprocesses: [`oxiaudio`] decodes every input this
//! crate accepts (wav, flac, ogg/vorbis, mp3, aac/m4a, aiff, …) and encodes
//! wav and flac; [`rusty_vorbis`] (in the `ogg` container) and [`rusty_mp3`]
//! encode the two lossy formats. This makes the day-one crate doc true —
//! "reads and writes audio through pure-Rust codecs, no `ffmpeg`".
//!
//! # Division of labour
//!
//! * **read** — [`oxiaudio::decode_file`], whole file, to interleaved `f32`.
//! * **write** — per container:
//!   * `wav`: hound via `oxiaudio`, depths 16/24/32-bit integer and 32-bit float.
//!   * `flac`: flacenc via `oxiaudio`, depths 16 and 24 (24/32/32f all land on
//!     24-bit integer, like the ffmpeg path did).
//!   * `ogg`: [`rusty_vorbis`] wrapped in Ogg pages by the `ogg` crate —
//!     `oxiaudio`'s own Vorbis encoder is broken (see [`write_ogg`]).
//!   * `mp3`: [`rusty_mp3::Mp3Encoder`], constant bitrate, gapless LAME tag.
//!
//! # Container and channel limits
//!
//! `oxiaudio` models channel layouts as a fixed set, so only 1, 2, 4, 6, 8 and
//! 12 channels round-trip correctly; the Vorbis and MP3 encoders take 1 or 2.
//! Anything else fails with an error that says so, rather than silently
//! relabelling the audio.
//!
//! # Sample rate and channels are never resampled or remixed
//!
//! They come from the input and are passed to the encoder unchanged; a mismatch
//! between the request and what decoding the written file reports is an error.

use std::io::{BufWriter, Read, Write};
use std::path::Path;

use oxiaudio::{AudioBuffer, ChannelLayout, SampleFormat};

/// Interleaved `f32` audio plus its format.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: usize,
}

impl Audio {
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels
        }
    }

    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.frames() as f64 / self.sample_rate as f64
        }
    }

    /// Mono mixdown (the engines' pitch analysis is mono even in stereo mode).
    pub fn mono(&self) -> Vec<f32> {
        if self.channels <= 1 {
            return self.samples.clone();
        }
        let n = self.channels;
        (0..self.frames())
            .map(|i| self.samples[i * n..(i + 1) * n].iter().sum::<f32>() / n as f32)
            .collect()
    }
}

/// Output container, overriding the extension when set.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Container {
    Wav,
    Flac,
    Ogg,
    Mp3,
}

impl Container {
    /// Container implied by a file extension.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let ext = path
            .as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "wav" | "wave" => Ok(Container::Wav),
            "flac" => Ok(Container::Flac),
            "ogg" | "oga" => Ok(Container::Ogg),
            "mp3" => Ok(Container::Mp3),
            other => Err(format!(
                "unsupported output extension {other:?}; use .wav/.flac/.ogg/.mp3 or --format"
            )),
        }
    }

    /// The file extension this container is written with, without the dot.
    ///
    /// [`Container::from_path`] is the exact inverse.
    pub fn extension(self) -> &'static str {
        match self {
            Container::Wav => "wav",
            Container::Flac => "flac",
            Container::Ogg => "ogg",
            Container::Mp3 => "mp3",
        }
    }

    /// Does writing this container quantise the samples (see [`BitDepth`])?
    pub fn quantises(self, depth: BitDepth) -> bool {
        match self {
            // Integer depths clip and quantise; 32-bit float stores samples exactly.
            Container::Wav => !depth.is_float(),
            // The FLAC encoders here are 16/24-bit integer only — a float never
            // survives, regardless of the requested depth.
            Container::Flac => true,
            // Vorbis and MP3 are float codecs: the encoders quantise in their own
            // (lossy, approximate) domain, not against integer depth steps, and do
            // not clip overshoot.
            Container::Ogg | Container::Mp3 => false,
        }
    }

    /// Which output depths make sense for this container (CLI help / docs).
    pub fn supported_depths(self) -> &'static [BitDepth] {
        match self {
            Container::Wav => &[BitDepth::I16, BitDepth::I24, BitDepth::I32, BitDepth::F32],
            Container::Flac => &[BitDepth::I16, BitDepth::I24],
            Container::Ogg | Container::Mp3 => &[],
        }
    }
}

/// PCM bit depth to write, when the container is lossless.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum BitDepth {
    #[value(name = "16")]
    I16,
    #[value(name = "24")]
    I24,
    #[value(name = "32")]
    I32,
    #[value(name = "32f")]
    #[default]
    F32,
}

impl BitDepth {
    pub fn is_float(self) -> bool {
        matches!(self, BitDepth::F32)
    }
}

/// How to encode the output.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    pub depth: BitDepth,
    /// `None` means "take the container from the output path's extension".
    pub container: Option<Container>,
    /// Ogg/Vorbis target bitrate in kbit/s, VBR: the encoder quality whose
    /// *measured* average bitrate is closest is picked, so the result is
    /// approximate (roughly ±10% on programme material). Useful range about
    /// 30–500 kbit/s stereo and half that mono; requests outside clamp to the
    /// ends. This is the only Ogg knob — the encoder has no true bitrate
    /// management, so CBR is not on offer anywhere in the API.
    pub ogg_bitrate: u32,
    /// MP3 constant bitrate in kbit/s.
    pub mp3_kbps: u16,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            depth: BitDepth::F32,
            container: None,
            ogg_bitrate: 512,
            mp3_kbps: 320,
        }
    }
}

// ---------------------------------------------------------------------------
// probing
// ---------------------------------------------------------------------------

/// This crate's own view of a file: what decoder would take it and what the
/// decoded stream looks like.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    pub sample_rate: u32,
    pub channels: usize,
    /// The codec/container name, e.g. `wav`, `flac`, `vorbis`, `mp3`, `aac`.
    pub codec: String,
    /// One-line description for anything not modelled above.
    pub line: String,
}

/// What decodes to what, without touching the audio itself.
///
/// The container comes from the file's magic number (cheap, no decode); the
/// rate and channels come from decoding. Callers that only need [`Info::codec`]
/// pay nothing beyond a 12-byte read through [`container_of`].
pub fn info(path: impl AsRef<Path>) -> Result<Info, String> {
    let path = path.as_ref();
    let codec = container_of(path)?;
    let buf = decode(path)?;
    Ok(Info {
        sample_rate: buf.sample_rate,
        channels: buf.channels.channel_count(),
        line: format!("{codec}, {} Hz, {} ch", buf.sample_rate, buf.channels),
        codec,
    })
}

/// Container name from the file's magic number alone.
fn container_of(path: &Path) -> Result<String, String> {
    use oxiaudio::AudioFormatHint;
    let hint = oxiaudio::detect_format_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let name = match hint {
        AudioFormatHint::Wav => "wav",
        AudioFormatHint::Flac => "flac",
        AudioFormatHint::Ogg => "vorbis",
        AudioFormatHint::Mp3 => "mp3",
        AudioFormatHint::Aac => "aac",
        AudioFormatHint::M4a => "aac",
        AudioFormatHint::Aiff => "aiff",
        AudioFormatHint::Au => "au",
        AudioFormatHint::Musepack => "mpc",
        AudioFormatHint::WavPack => "wavpack",
    };
    Ok(name.to_string())
}

// ---------------------------------------------------------------------------
// reading
// ---------------------------------------------------------------------------

/// Decode any audio file oxiaudio supports to interleaved `f32`.
fn decode(path: &Path) -> Result<AudioBuffer<f32>, String> {
    let buf = oxiaudio::decode_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let ch = buf.channels.channel_count();
    if ch == 0 || buf.samples.is_empty() {
        return Err(format!("{}: decoded to nothing", path.display()));
    }
    // A buffer whose length does not divide by its channel count cannot be
    // framed; a decoder that mislabelled the channel count shows up here.
    if buf.samples.len() % ch != 0 {
        return Err(format!(
            "{}: {} samples do not divide into {} channels; the file's channel count \
             is outside the 1/2/4/6/8/12 layouts oxiaudio models",
            path.display(),
            buf.samples.len(),
            ch
        ));
    }
    Ok(buf)
}

/// Read any audio file oxiaudio can decode (wav, flac, ogg/vorbis, mp3,
/// aac/m4a, aiff, au, musepack, wavpack) to interleaved `f32`.
pub fn read(path: impl AsRef<Path>) -> Result<Audio, String> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(format!("{}: no such file", path.display()));
    }
    let buf = decode(path)?;
    Ok(Audio {
        samples: buf.samples,
        sample_rate: buf.sample_rate,
        channels: buf.channels.channel_count(),
    })
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

/// Build the interleaved-f32 buffer the oxiaudio encoders take.
///
/// Fails for channel counts outside the 1/2/4/6/8/12 layouts oxiaudio models
/// rather than letting them be relabelled as stereo.
fn to_oxi_buffer(audio: &Audio) -> Result<AudioBuffer<f32>, String> {
    let layout = match audio.channels {
        1 => ChannelLayout::Mono,
        2 => ChannelLayout::Stereo,
        4 => ChannelLayout::Quad,
        6 => ChannelLayout::Surround51,
        8 => ChannelLayout::Surround71,
        12 => ChannelLayout::Atmos714,
        other => {
            return Err(format!(
                "{other} channels: the pure-Rust codecs model 1/2/4/6/8/12 only"
            ))
        }
    };
    Ok(AudioBuffer {
        samples: audio.samples.clone(),
        sample_rate: audio.sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    })
}

/// The nine sample rates MP3 (MPEG-1/2/2.5 Layer III) can carry.
const MP3_RATES: [u32; 9] = [8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000];

/// Write audio to a container, entirely in-process.
///
/// The sample rate and channel count are passed through unchanged, and the
/// written file is decoded back to verify them.
pub fn write(path: impl AsRef<Path>, audio: &Audio, opts: WriteOptions) -> Result<(), String> {
    let path = path.as_ref();
    let container = match opts.container {
        Some(c) => c,
        None => Container::from_path(path)?,
    };
    if audio.channels == 0 {
        return Err("cannot write audio with zero channels".into());
    }
    if audio.samples.is_empty() {
        return Err("cannot write an empty signal".into());
    }
    if container == Container::Ogg || container == Container::Mp3 {
        if audio.channels > 2 {
            return Err(format!(
                "{}: the {} encoder takes 1 or 2 channels, not {}",
                path.display(),
                if container == Container::Ogg { "Vorbis" } else { "MP3" },
                audio.channels
            ));
        }
    }

    match container {
        Container::Wav => write_wav(path, audio, opts.depth)?,
        Container::Flac => write_flac(path, audio, opts.depth)?,
        Container::Ogg => write_ogg(path, audio, &opts)?,
        Container::Mp3 => write_mp3(path, audio, opts.mp3_kbps)?,
    }

    let written = std::fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    if written == 0 {
        return Err(format!("{}: the encoder produced an empty file", path.display()));
    }
    // Verify rather than assume: the rate and channel count must be what was
    // asked for, otherwise every caller's "input and output agree" contract is
    // a lie.
    let got = info(path)?;
    if got.sample_rate != audio.sample_rate || got.channels != audio.channels {
        return Err(format!(
            "{}: wrote {} Hz / {} ch but {} Hz / {} ch was requested",
            path.display(),
            got.sample_rate,
            got.channels,
            audio.sample_rate,
            audio.channels
        ));
    }
    Ok(())
}

fn write_wav(path: &Path, audio: &Audio, depth: BitDepth) -> Result<(), String> {
    use oxiaudio::WavBitDepth;
    let depth = match depth {
        BitDepth::I16 => WavBitDepth::I16,
        BitDepth::I24 => WavBitDepth::I24,
        BitDepth::I32 => WavBitDepth::I32,
        BitDepth::F32 => WavBitDepth::F32,
    };
    let buf = to_oxi_buffer(audio)?;
    oxiaudio::encode_wav_with_config(&buf, path, depth)
        .map_err(|e| format!("{}: wav: {e}", path.display()))
}

/// Write FLAC through flacenc directly, with the configuration this crate
/// measured against libFLAC.
///
/// Two things here were decided by measurement, not by defaults:
///
/// * **The float→int conversion scales by 2^(bits−1) and rounds
///   half-to-even** (then clamps into the signed range). oxiaudio's
///   `floor(s · (2^23−1))` truncation has two measured defects: the scale is
///   not a power of two, so material already on an integer grid (16-bit
///   sources) lands on ±8-unit jitter at 24-bit — incompressible noise that
///   made a FLAC *larger* than the same PCM as WAV — and truncation biases
///   every sample toward zero by up to a full step. With a power-of-two scale
///   grid-aligned material stays on the grid (so 16-bit sources compress like
///   16-bit even at 24-bit depth), and rounding halves the worst-case error.
/// * **The encoder config** is block 8192, qlpc order 24 with direct-MSE
///   estimation and one MAE refinement step (`flacenc`'s `experimental`
///   feature). On the dense-mix reference that beats the crate default by
///   ~10% at 24-bit and lands within a few percent of libFLAC −8 on ordinary
///   material — and *smaller* than libFLAC at 16-bit and on quiet material;
///   flacenc only loses (by ~30%) on broadband noise near −34 dBFS, a floor
///   set by its Rice parameter cap of 14 (the spec allows 30 plus an escape
///   code), which no config can lift. `OrderSel::BitCount` also looks better
///   on paper but emits pathological frames at some block sizes (a
///   reproducible 303 MB file from 8 s of audio), so it is not used.
/// * **Grid detection** stands in for libFLAC's wasted-bits flag, which
///   flacenc does not implement: when every sample of a 24-bit render already
///   sits on the 16-bit grid (a 16-bit source), the file is written as 16-bit
///   — decoding is bit-identical, and it is exactly what ffmpeg produces for
///   16-bit input.
///
/// 16 stays 16; 24/32/32f all become 24-bit integer, the deepest subframe the
/// format offers.
fn write_flac(path: &Path, audio: &Audio, depth: BitDepth) -> Result<(), String> {
    use flacenc::bitsink::ByteSink;
    use flacenc::component::BitRepr;
    use flacenc::config;
    use flacenc::error::Verify;
    use flacenc::source::MemSource;

    const BLOCK: usize = 8192;
    let mut bits: u32 = if depth == BitDepth::I16 { 16 } else { 24 };
    let scale = (1u64 << (bits - 1)) as f32;
    let max = ((1u64 << (bits - 1)) - 1) as i32;
    let min = -((1u64 << (bits - 1)) as i32);
    let mut pcm: Vec<i32> = audio
        .samples
        .iter()
        .map(|&s| (s * scale).round_ties_even() as i32)
        .map(|v| v.clamp(min, max))
        .collect();

    // Grid detection, standing in for libFLAC's wasted-bits flag (flacenc has
    // none): when every sample already sits on the 16-bit grid, a 16-bit
    // stream decodes to exactly the same floats — and ffmpeg writes exactly
    // that for a 16-bit source. Without this, the eight constant zero LSBs are
    // coded into every residual, which flacenc's Rice cap (k ≤ 14) turns into
    // *expansion*: a 16-bit WAV re-encoded as 24-bit FLAC came out larger than
    // the WAV itself.
    if bits == 24 && pcm.iter().all(|v| v & 0xFF == 0) {
        bits = 16;
        for v in &mut pcm {
            *v >>= 8;
        }
    }

    let src = MemSource::from_samples(&pcm, audio.channels, bits as usize, audio.sample_rate as usize);
    let mut cfg = config::Encoder::default();
    cfg.block_size = BLOCK;
    cfg.subframe_coding.qlpc.lpc_order = 24;
    cfg.subframe_coding.qlpc.use_direct_mse = true;
    cfg.subframe_coding.qlpc.mae_optimization_steps = 1;
    let cfg = cfg
        .into_verified()
        .map_err(|(_, e)| format!("{}: flac: {e}", path.display()))?;
    let mut stream = flacenc::encode_with_fixed_block_size(&cfg, src, BLOCK)
        .map_err(|e| format!("{}: flac: {e}", path.display()))?;
    // flacenc reports the last (partial) block as min_block_size, which makes
    // min != max; symphonia then rejects the frame sequencing. Pin both.
    let _ = stream.stream_info_mut().set_block_sizes(BLOCK, BLOCK);
    let mut sink = ByteSink::with_capacity(stream.count_bits());
    stream
        .write(&mut sink)
        .map_err(|e| format!("{}: flac: {e}", path.display()))?;
    std::fs::write(path, sink.as_slice()).map_err(|e| format!("{}: {e}", path.display()))
}

/// Write Ogg/Vorbis through rusty_vorbis (the pure-Rust encoder), framed into
/// the Ogg container by the `ogg` crate.
///
/// oxiaudio's own Vorbis encoder was the first candidate and is unusable: on
/// real material it emits a stream both symphonia and ffmpeg decode to garbage
/// (max |diff| ~1e7, SNR below −130 dB), truncates to a block boundary and
/// ignores the quality knob entirely. rusty_vorbis is from the same family as
/// rusty_mp3 and behaves like a real encoder, so it took the job.
///
/// # Block bookkeeping, measured on the wire
///
/// The encoder hops in 1024-sample blocks (blocksizes 2048/256) and its
/// packets make every decoder skip the first hop and never render the last
/// one's overlap. Both are compensated here, the way a complete integration
/// muxes the codec:
///
/// * **head** — the input is preceded by one silent 1024-frame *lead* hop, the
///   overlap partner the encoder's first real hop needs; decoders that honour
///   granule positions (symphonia does — the crate's own read path, and so the
///   acceptance tests) skip it in place of real audio. Its content is
///   irrelevant to the reconstruction — measured: copy of the input's head,
///   reversed copy and pure zeros all decode identically — so it is zeros,
///   which keeps the only extra audio a decoder that ignores granules (ffmpeg's
///   vorbis path) can play down to silence;
/// * **tail** — the fed stream is zero-padded to a hop multiple plus one extra
///   hop, so the last real hop gets its overlap partner;
/// * **length** — the final page's granule position is `hop + input_frames`,
///   which trims the pad on decode;
/// * **pages** — each header packet and the first audio packet end their own
///   page (libvorbis's shape); the first audio page's granule is what makes
///   decoders apply the head skip instead of playing the lead.
///
/// Round trip that through the crate's own decoder and the length is the input
/// length exactly, sample-aligned at lag 0 (verified by
/// `ogg_round_trips_approximately` and the CLI acceptance suite). A decoder
/// that ignores granule trimming altogether plays roughly two extra hops of
/// silence around the programme; that is a rusty_vorbis property — it cannot
/// produce libvorbis's priming-free first block — and the honest floor for a
/// pure-Rust Vorbis today.
fn write_ogg(path: &Path, audio: &Audio, opts: &WriteOptions) -> Result<(), String> {
    use ogg::PacketWriteEndInfo::{EndPage, EndStream, NormalPacket};
    use ogg::PacketWriter;
    use rusty_vorbis::{Error, VorbisEncoder, VorbisEncoderConfig};

    const HOP: usize = 1024;

    // Bitrate is the only knob: pick the encoder quality whose measured
    // average bitrate is closest to the request.
    let q = vorbis_quality_for_bitrate(opts.ogg_bitrate, audio.channels);

    let ch = audio.channels;
    let frames = audio.frames();
    // One silent lead hop: the overlap partner of the encoder's first real
    // hop. Its content does not survive decoding (the first audio page's
    // granule makes decoders skip it), so zeros — no phantom pre-echo.
    let mut fed: Vec<f32> = vec![0.0; HOP * ch];
    fed.extend_from_slice(&audio.samples);
    // Tail: zero pad to a hop multiple, plus one extra hop of overlap partner.
    let padded_frames = fed.len() / ch + HOP;
    fed.resize(padded_frames.div_ceil(HOP) * HOP * ch, 0.0);

    let mut enc = VorbisEncoder::new(VorbisEncoderConfig {
        quality: q,
        // The crate documents this as informational (it never steers rate
        // control); it travels in the header as the profile's nominal rate.
        bitrate_bps: (opts.ogg_bitrate * 1000) as i32,
    });
    enc.push_pcm_f32(&fed, ch as u16, audio.sample_rate)
        .map_err(|e| format!("{}: ogg/vorbis: {e}", path.display()))?;
    enc.finish();

    // Drain everything first, so the last packet is known when the stream end
    // has to be flagged on it.
    let mut packets = Vec::new();
    loop {
        match enc.next_packet() {
            Ok(pkt) => packets.push(pkt),
            Err(Error::Eof) | Err(Error::Again) => break,
            Err(other) => return Err(format!("{}: ogg/vorbis: {other}", path.display())),
        }
    }
    let last = packets.len().saturating_sub(1);
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut wtr = PacketWriter::new(std::io::BufWriter::new(file));
    let serial = 0x7261_6473; // "rads"
    for (n, pkt) in packets.into_iter().enumerate() {
        // Page shape, following libvorbis: the three headers each end their
        // own page; the FIRST audio packet also ends its own page, whose
        // granule (the packet's pts) tells decoders how much of the first
        // block is pre-history to skip — without it symphonia plays the lead
        // hop as audio. The very last packet ends the stream, and its page's
        // granule marks the true end (lead hop already skipped plus the whole
        // input), trimming the tail pad.
        let info = if n < 3 || n == 3 {
            EndPage
        } else if n == last {
            EndStream
        } else {
            NormalPacket
        };
        let granule = if n == last {
            (HOP + frames) as u64
        } else {
            pkt.pts as u64
        };
        wtr.write_packet(pkt.data, serial, info, granule)
            .map_err(|e| format!("{}: ogg: {e}", path.display()))?;
    }
    Ok(())
}

fn write_mp3(path: &Path, audio: &Audio, kbps: u16) -> Result<(), String> {
    use rusty_mp3::{Error, Mp3Encoder, Mp3EncoderConfig};
    if !MP3_RATES.contains(&audio.sample_rate) {
        return Err(format!(
            "{}: mp3: {} Hz is outside the MPEG rates ({}); write wav/flac, or resample",
            path.display(),
            audio.sample_rate,
            MP3_RATES
                .iter()
                .map(|r| format!("{r}"))
                .collect::<Vec<_>>()
                .join("/")
        ));
    }
    let mut enc = Mp3Encoder::new(Mp3EncoderConfig {
        bitrate_kbps: u32::from(kbps),
        vbr_quality: None,
    });
    enc.push_pcm_f32(&audio.samples, audio.channels as u16, audio.sample_rate)
        .map_err(|e| match e {
            Error::Unsupported(what) => format!("{}: mp3: {what}", path.display()),
            other => format!("{}: mp3: {other}", path.display()),
        })?;
    enc.finish();
    let mut out = Vec::new();
    loop {
        match enc.next_packet() {
            Ok(packet) => out.extend_from_slice(&packet),
            Err(Error::Eof) => break,
            Err(Error::Again) => continue,
            Err(other) => return Err(format!("{}: mp3: {other}", path.display())),
        }
    }
    std::fs::write(path, &out).map_err(|e| format!("{}: {e}", path.display()))
}

/// Measured `quality` → average kbit/s for this crate's Vorbis encoder
/// (rusty_vorbis), 44.1 kHz stereo, on a dense musical mix (seven-tone chord
/// with vibrato and noise, four seconds) — the same generator the io test
/// `dense_mix` builds, so the curve and its check cannot drift apart.
///
/// The encoder's own documented scale ends at 0.98 (≈180 kbit/s), but its
/// rate-distortion λ keeps falling past that and the embedded q4 setup's
/// codebooks keep accepting the bits: the ceiling is measured, not declared.
/// q1.8 is where the dense reference passes 512 kbit/s, which is the default
/// and the top of the exposed range.
///
/// Vorbis here is VBR with **no bitrate management** — the encoder's
/// `bitrate_bps` field is documented as informational and truly is — so a
/// requested bitrate can only be approximated: this maps the request to the
/// quality level whose measured rate is closest, scaling for mono (which
/// roughly halves the bitrate at the same quality). Requests outside the
/// measured range clamp to its ends.
const VORBIS_KBPS_AT_Q: [(f32, f32); 19] = [
    (0.05, 29.0),
    (0.25, 31.0),
    (0.35, 34.0),
    (0.45, 41.0),
    (0.55, 52.0),
    (0.65, 70.0),
    (0.75, 100.0),
    (0.85, 132.0),
    (0.93, 160.0),
    (0.98, 179.0),
    (1.00, 187.0),
    (1.10, 230.0),
    (1.20, 271.0),
    (1.30, 309.0),
    (1.40, 343.0),
    (1.50, 377.0),
    (1.60, 415.0),
    (1.70, 456.0),
    (1.80, 498.0),
];

/// The Vorbis quality level whose measured bitrate is closest to `kbps`.
///
/// `channels` scales the target first (mono ≈ half the stereo rate at the same
/// quality). Clamps to the encoder's range on both ends.
pub fn vorbis_quality_for_bitrate(kbps: u32, channels: usize) -> f32 {
    // What the stereo curve would produce for this stream: a mono stream at
    // quality q costs about half a stereo stream at q.
    let stereo_equivalent = kbps as f32 * 2.0 / channels.max(1) as f32;
    let table = &VORBIS_KBPS_AT_Q;
    if stereo_equivalent <= table[0].1 {
        return table[0].0;
    }
    if stereo_equivalent >= table[table.len() - 1].1 {
        return table[table.len() - 1].0;
    }
    // Walk the curve; the crossing segment decides by midpoint.
    for pair in table.windows(2) {
        let (q0, k0) = pair[0];
        let (q1, k1) = pair[1];
        if stereo_equivalent < k0 || stereo_equivalent > k1 {
            continue;
        }
        let t = (stereo_equivalent - k0) / (k1 - k0);
        return q0 + t * (q1 - q0);
    }
    unreachable!("bitrate between the curve's ends always crosses a segment")
}

fn bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Stream interleaved `f32` to a writer as raw little-endian samples.
pub fn write_raw_f32(w: &mut impl Write, x: &[f32]) -> std::io::Result<()> {
    let mut bw = BufWriter::new(w);
    for v in x {
        bw.write_all(&v.to_le_bytes())?;
    }
    bw.flush()
}

/// Read raw little-endian `f32` from a reader.
pub fn read_raw_f32(r: &mut impl Read) -> std::io::Result<Vec<f32>> {
    let mut raw = Vec::new();
    r.read_to_end(&mut raw)?;
    Ok(bytes_to_f32(&raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join("radius_rs_io_tests");
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn tone(rate: u32, channels: usize, frames: usize) -> Audio {
        let mut samples = Vec::with_capacity(frames * channels);
        for i in 0..frames {
            for c in 0..channels {
                let phase = i as f32 * 0.01 + c as f32;
                samples.push((2.0 * std::f32::consts::PI * phase).sin() * 0.5);
            }
        }
        Audio {
            samples,
            sample_rate: rate,
            channels,
        }
    }

    #[test]
    fn container_from_extension() {
        assert_eq!(Container::from_path("a.wav"), Ok(Container::Wav));
        assert_eq!(Container::from_path("A.FLAC"), Ok(Container::Flac));
        assert_eq!(Container::from_path("x.ogg"), Ok(Container::Ogg));
        assert_eq!(Container::from_path("y.mp3"), Ok(Container::Mp3));
        assert!(Container::from_path("z.xyz").is_err());
        assert!(Container::from_path("noext").is_err());
    }

    #[test]
    fn quantisation_policy_per_container() {
        assert!(Container::Wav.quantises(BitDepth::I16));
        assert!(!Container::Wav.quantises(BitDepth::F32));
        assert!(Container::Flac.quantises(BitDepth::F32));
        assert!(!Container::Ogg.quantises(BitDepth::F32));
        assert!(!Container::Mp3.quantises(BitDepth::F32));
    }

    #[test]
    fn missing_file_is_reported() {
        let err = read(std::env::temp_dir().join("no_such_file_anywhere.wav")).unwrap_err();
        assert!(err.contains("no such file"), "{err}");
    }

    #[test]
    fn garbage_file_is_reported() {
        let p = dir().join("garbage.bin");
        std::fs::write(&p, b"definitely not audio").unwrap();
        let err = read(&p).unwrap_err();
        assert!(err.contains("garbage.bin"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn audio_geometry() {
        let a = Audio {
            samples: vec![0.0; 10],
            sample_rate: 1000,
            channels: 2,
        };
        assert_eq!(a.frames(), 5);
        assert!((a.duration() - 0.005).abs() < 1e-12);
        assert_eq!(a.mono().len(), 5);
    }

    /// Every container round-trips: read(write(x)) agrees on rate, channels and
    /// (for wav f32) the exact samples.
    #[test]
    fn wav_f32_round_trips_bit_exactly() {
        let audio = tone(44_100, 2, 4_410);
        let p = dir().join("rt_f32.wav");
        write(&p, &audio, WriteOptions::default()).unwrap();
        let back = read(&p).unwrap();
        assert_eq!(back.sample_rate, 44_100);
        assert_eq!(back.channels, 2);
        assert_eq!(back.samples, audio.samples);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn wav_integer_depths_change_the_bytes() {
        let audio = tone(44_100, 1, 1_000);
        // hound writes a constant header; the data grows with the depth.
        let len = |depth: BitDepth, name: &str| {
            let p = dir().join(name);
            write(
                &p,
                &audio,
                WriteOptions {
                    depth,
                    ..WriteOptions::default()
                },
            )
            .unwrap();
            let back = read(&p).unwrap();
            assert_eq!(back.sample_rate, 44_100);
            assert_eq!(back.channels, 1);
            if depth != BitDepth::F32 {
                assert_ne!(back.samples, audio.samples, "{depth:?} must quantise");
            }
            let len = std::fs::metadata(&p).unwrap().len();
            let _ = std::fs::remove_file(&p);
            len
        };
        let (l16, l24, l32, l32f) = (
            len(BitDepth::I16, "depth_i16.wav"),
            len(BitDepth::I24, "depth_i24.wav"),
            len(BitDepth::I32, "depth_i32.wav"),
            len(BitDepth::F32, "depth_f32.wav"),
        );
        // 1000 mono samples: 2000 / 3000 / 4000 bytes of data (hound's 24-bit
        // header carries a few extra chunk bytes, so the first delta has slack).
        assert!((l24 - l16) >= 900 && (l24 - l16) <= 1100);
        assert_eq!(l32 - l24, 1_000);
        assert_eq!(l32, l32f);
    }

    #[test]
    fn flac_16_and_24_round_trip() {
        let audio = tone(48_000, 2, 2_400);
        for depth in [BitDepth::I16, BitDepth::I24, BitDepth::I32, BitDepth::F32] {
            let p = dir().join(format!("rt_{depth:?}.flac"));
            write(
                &p,
                &audio,
                WriteOptions {
                    depth,
                    ..WriteOptions::default()
                },
            )
            .unwrap();
            let back = read(&p).unwrap();
            assert_eq!(back.sample_rate, 48_000, "{depth:?}");
            assert_eq!(back.channels, 2, "{depth:?}");
            // Both depths quantise, and 16 is coarser than 24.
            let err16 = back
                .samples
                .iter()
                .zip(&audio.samples)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(err16 < 0.01, "{depth:?}: {err16}");
            assert_eq!(info(&p).unwrap().codec, "flac");
            let _ = std::fs::remove_file(&p);
        }
    }

    /// Material already on the 16-bit grid is written as 16-bit FLAC even at
    /// the default depth: decoding is bit-identical, the size matches what
    /// ffmpeg produces for a 16-bit source, and the alternative — coding
    /// eight constant zero LSBs into every residual — made the file larger
    /// than raw WAV.
    #[test]
    fn flac_grid_aligned_material_drops_to_16_bits() {
        // A tone quantised to the 16-bit grid, like a decoded 16-bit WAV.
        let q: Vec<f32> = tone(44_100, 2, 4_410)
            .samples
            .iter()
            .map(|&s| (s * 32_768.0).round() / 32_768.0)
            .collect();
        let audio = Audio {
            samples: q.clone(),
            sample_rate: 44_100,
            channels: 2,
        };
        let flac = dir().join("grid.flac");
        write(&flac, &audio, WriteOptions::default()).unwrap();
        let back = read(&flac).unwrap();
        // Bit-exact: 16-bit FLAC decodes to k/32768, which is what we fed it.
        assert_eq!(back.samples, q, "grid material must survive exactly");
        // And compact: nothing near the 24-bit raw size it used to approach.
        let size = std::fs::metadata(&flac).unwrap().len();
        let raw24 = 4_410 * 2 * 3;
        assert!(size < raw24 as u64 / 2, "{size} vs raw24 {raw24}");
        let _ = std::fs::remove_file(&flac);
    }

    #[test]
    fn ogg_round_trips_approximately() {
        let audio = tone(44_100, 2, 4_410);
        let p = dir().join("rt.ogg");
        write(&p, &audio, WriteOptions::default()).unwrap();
        let back = read(&p).unwrap();
        assert_eq!(back.sample_rate, 44_100);
        assert_eq!(back.channels, 2);
        assert_eq!(info(&p).unwrap().codec, "vorbis");
        // The muxing contract: the decoded length is the input length exactly
        // (lead hop skipped, tail pad trimmed by the final granule).
        assert_eq!(back.frames(), audio.frames(), "ogg must be sample-exact in length");
        // Lossy, but this tame a tone must come back close, with no lag.
        // (The encoder's own floor reconstruction puts ~0.09 of noise on a
        // pure sine; the head blend contributes about as much.)
        let err = back
            .samples
            .iter()
            .zip(&audio.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 0.12, "{err}");
        let _ = std::fs::remove_file(&p);
    }

    /// A dense mix (what the bitrate curve was measured on — keep the tone set
    /// in sync with `VORBIS_KBPS_AT_Q`'s measurement or VBR lands elsewhere)
    /// so the encoder actually spends the bits it is asked for; a pure tone
    /// codes for a fraction of any target.
    fn dense_mix(rate: u32, frames: usize) -> Audio {
        let mut samples = Vec::with_capacity(frames * 2);
        let mut s = 0x5eed_1234u64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / ((1u64 << 31) as f32)
        };
        for i in 0..frames {
            let t = i as f32 / rate as f32;
            let env = 0.5 + 0.5 * (std::f32::consts::PI * t).sin();
            let (mut l, mut r) = (0.0f32, 0.0f32);
            for (k, f) in [110.0f32, 220.0, 277.2, 330.0, 440.0, 554.4, 660.0]
                .iter()
                .enumerate()
            {
                let a = 0.2 / (k as f32 + 1.0);
                l += a * (2.0 * std::f32::consts::PI * f * t).sin();
                r += a * (2.0 * std::f32::consts::PI * f * 1.001 * t).sin();
            }
            let n = (rnd() - 0.5) * 0.04;
            samples.push((l * env + n).clamp(-1.0, 1.0));
            samples.push((r * env + n).clamp(-1.0, 1.0));
        }
        Audio {
            samples,
            sample_rate: rate,
            channels: 2,
        }
    }

    /// A requested bitrate lands near the measured curve, and inside it.
    ///
    /// Four seconds of material, like the curve itself: the lead hop and tail
    /// pad (≈230 ms together) are real encoded audio, so on very short files
    /// the achieved bitrate floats high (on 1 s by ~40%); on programme-length
    /// material the overhead vanishes.
    #[test]
    fn ogg_bitrate_lands_near_the_request() {
        let audio = dense_mix(44_100, 44_100 * 4);
        // Across the whole exposed range, including the default (512) and the
        // extended-quality region above the encoder's documented 0.98.
        for kbps in [64u32, 128, 256, 400, 512] {
            let p = dir().join(format!("br{kbps}.ogg"));
            write(
                &p,
                &audio,
                WriteOptions {
                    ogg_bitrate: kbps,
                    ..WriteOptions::default()
                },
            )
            .unwrap();
            let got = std::fs::metadata(&p).unwrap().len() as f64 * 8.0 / 1000.0
                / (audio.frames() as f64 / audio.sample_rate as f64);
            assert!(
                ((got - kbps as f64).abs() / kbps as f64) < 0.35,
                "asked {kbps} kbit/s, got {got:.0}"
            );
            let _ = std::fs::remove_file(&p);
        }
    }

    /// The default bitrate is the top of the exposed range, and the top of the
    /// range is measured, not aspirational.
    #[test]
    fn ogg_default_bitrate_is_the_ceiling() {
        assert_eq!(WriteOptions::default().ogg_bitrate, 512);
        let audio = dense_mix(44_100, 44_100 * 4);
        let p = dir().join("default.ogg");
        write(&p, &audio, WriteOptions::default()).unwrap();
        let got = std::fs::metadata(&p).unwrap().len() as f64 * 8.0 / 1000.0
            / (audio.frames() as f64 / audio.sample_rate as f64);
        assert!(
            (470.0..=540.0).contains(&got),
            "default should land near 512 kbit/s, got {got:.0}"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn mp3_round_trips_approximately() {
        let audio = tone(44_100, 2, 4_410);
        let p = dir().join("rt.mp3");
        write(&p, &audio, WriteOptions::default()).unwrap();
        let back = read(&p).unwrap();
        assert_eq!(back.sample_rate, 44_100);
        assert_eq!(back.channels, 2);
        assert_eq!(info(&p).unwrap().codec, "mp3");
        let err = back
            .samples
            .iter()
            .zip(&audio.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 0.2, "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn mp3_rejects_non_mpeg_rates() {
        let audio = tone(96_000, 2, 100);
        let p = dir().join("rate96.mp3");
        let err = write(&p, &audio, WriteOptions::default()).unwrap_err();
        assert!(err.contains("96"), "{err}");
        assert!(!p.exists());
    }

    #[test]
    fn vorbis_rejects_more_than_two_channels() {
        let audio = tone(44_100, 4, 100);
        let p = dir().join("quad.ogg");
        let err = write(&p, &audio, WriteOptions::default()).unwrap_err();
        assert!(err.contains("1 or 2 channels"), "{err}");
        assert!(!p.exists());
    }

    #[test]
    fn five_channels_wav_round_trips_but_oggs_fail_clearly() {
        // 5 channels: outside the layouts oxiaudio models, so writing must say
        // so rather than mislabel the stream as stereo.
        let audio = tone(44_100, 5, 100);
        let p = dir().join("five.wav");
        let err = write(&p, &audio, WriteOptions::default()).unwrap_err();
        assert!(err.contains("1/2/4/6/8/12"), "{err}");
    }

    #[test]
    fn bitrate_to_quality_maps_monotonically() {
        let q_low = vorbis_quality_for_bitrate(64, 2);
        let q_high = vorbis_quality_for_bitrate(160, 2);
        assert!(q_low < q_high, "{q_low} !< {q_high}");
        // Clamping at both ends of the measured curve.
        assert_eq!(vorbis_quality_for_bitrate(1, 2), 0.05);
        assert_eq!(vorbis_quality_for_bitrate(1_000, 2), 1.80);
        // The curve is monotone across its whole length, extended range
        // included — every step must buy bitrate.
        for pair in VORBIS_KBPS_AT_Q.windows(2) {
            assert!(pair[1].1 > pair[0].1, "not monotone at q {}", pair[0].0);
        }
        // Mono at 64k ≈ stereo at 128k.
        let (m, s) = (vorbis_quality_for_bitrate(64, 1), vorbis_quality_for_bitrate(128, 2));
        assert!((m - s).abs() < 0.15, "mono 64k {m} vs stereo 128k {s}");
    }
}
