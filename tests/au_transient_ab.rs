//! Transient-sharpness A/B against the user's actual Audition renders:
//! envelope correlation overall and specifically in attack regions, with the
//! transient subsystem off vs on (the shipped default).
//!
//! `#[ignore]`d: needs the `.local-testdata` corpus and runs full-length
//! vocoder renders.

#![allow(dead_code)]

mod common;

use common::{acceptance_audio_or_skip, audition_reference_named, mono, read_wav, root};
use radius_rs::vocoder::VocoderState;

/// 10 ms peak envelope (mono).
fn env(x: &[f32], rate: u32) -> Vec<f64> {
    let hop = (rate as usize / 100).max(1);
    let m: Vec<f64> = x.iter().map(|v| f64::from(*v).abs()).collect();
    m.chunks(hop)
        .map(|c| c.iter().fold(0.0f64, |a, v| a.max(*v)))
        .collect()
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (ma, mb) = (
        a[..n].iter().sum::<f64>() / n as f64,
        b[..n].iter().sum::<f64>() / n as f64,
    );
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..n {
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma).powi(2);
        db += (b[i] - mb).powi(2);
    }
    num / (da * db).sqrt()
}

/// Attack regions: the envelope indices with the largest positive 3-bin
/// rise, thinned to ≥ 20 bins (200 ms) apart, top 64.
fn attack_centers(e: &[f64]) -> Vec<usize> {
    let mut rises: Vec<(f64, usize)> = (1..e.len().saturating_sub(1))
        .map(|i| (e[i + 1] - e[i - 1], i))
        .collect();
    rises.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut picked: Vec<usize> = Vec::new();
    for (_, i) in rises {
        if picked.iter().all(|p| (*p as isize - i as isize).abs() >= 20) {
            picked.push(i);
            if picked.len() >= 64 {
                break;
            }
        }
    }
    picked.sort_unstable();
    picked
}

fn report(tag: &str, got: &[f32], want: &[f32], rate: u32) {
    let a = mono(got, 2);
    let b = mono(want, 2);
    let n = a.len().min(b.len());
    let (ea, eb) = (env(&a[..n], rate), env(&b[..n], rate));
    let (overall, lag) = best_corr(&ea, &eb, 30);
    // Per-attack local correlation at the global best lag, ±30 bins.
    let mut local = Vec::new();
    for &c in &attack_centers(&eb) {
        let lo = c.saturating_sub(30);
        let hi = (c + 30).min(ea.len()).min(eb.len());
        if hi + 30 <= ea.len() && hi - lo >= 10 {
            let l = (lo as isize + lag as isize).max(0) as usize;
            let h = (hi as isize + lag as isize).max(0) as usize;
            if h > l && h <= ea.len() {
                local.push(corr(&ea[l..h], &eb[lo..hi]));
            }
        }
    }
    let att_mean = local.iter().sum::<f64>() / local.len().max(1) as f64;
    // Attack sharpness ratio: mean (rise height) ours / AU's at the AU
    // onsets, each at its own best lag within ±3 bins (the re-timing can
    // shift a region by a bin or two).
    let mut sharp = 0.0f64;
    let mut cnt = 0usize;
    for &c in &attack_centers(&eb) {
        let hi = (c + 2).min(eb.len().saturating_sub(1));
        let rise_b = eb[hi] - eb[c.saturating_sub(1)];
        if rise_b <= 1e-4 {
            continue;
        }
        let mut best = f64::NEG_INFINITY;
        for dl in -3isize..=3 {
            let ci = (c as isize + dl).max(0) as usize;
            let hi_a = (ci + 2).min(ea.len().saturating_sub(1));
            if hi_a > ci.saturating_sub(1) && hi_a < ea.len() {
                let r = (ea[hi_a] - ea[ci.saturating_sub(1)]) / rise_b;
                if r > best {
                    best = r;
                }
            }
        }
        if best.is_finite() {
            sharp += best;
            cnt += 1;
        }
    }
    let sharp = sharp / cnt.max(1) as f64;
    let ms = lag * 10;
    println!(
        "{tag}: contour corr {overall:.4} (lag {lag:+} bins = {ms:+} ms), \
         attack-window corr {att_mean:.4}, attack-rise ratio ours/AU {sharp:.3} \
         ({} regions)",
        local.len(),
    );
}

/// Best correlation over ±`span` bin lag: (corr, lag applied to `a`).
fn best_corr(a: &[f64], b: &[f64], span: isize) -> (f64, isize) {
    let n = a.len().min(b.len());
    let mut best = (f64::NEG_INFINITY, 0isize);
    for lag in -span..=span {
        let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
        let mut cnt = 0;
        let (ma, mb) = (mean(a), mean(b));
        for i in 0..n {
            let j = i as isize + lag;
            if j < 0 || j as usize >= n {
                continue;
            }
            num += (a[j as usize] - ma) * (b[i] - mb);
            da += (a[j as usize] - ma).powi(2);
            db += (b[i] - mb).powi(2);
            cnt += 1;
        }
        if cnt > 10 {
            let c = num / (da * db).sqrt();
            if c > best.0 {
                best = (c, lag);
            }
        }
    }
    best
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}

#[test]
#[ignore = "full-length renders against the AU corpus (~4 min)"]
fn au_transient_ab() {
    let Some(input_path) = acceptance_audio_or_skip("au_transient_ab") else {
        return;
    };
    let input = read_wav(&input_path);
    for name in ["au-vc+3.wav", "au-vc+3_200.wav", "au-vc+3_50.wav"] {
        let Some(path) = audition_reference_named(name) else {
            eprintln!("SKIP au_transient_ab: no {name}");
            continue;
        };
        let want = read_wav(&path).samples;
        let semis = 3.0;
        let tempo = if name.ends_with("_200.wav") {
            200.0
        } else if name.ends_with("_50.wav") {
            50.0
        } else {
            100.0
        };
        let target = (input.frames() as f64 * tempo / 100.0).round() as usize;

        let mut off = VocoderState::new(input.rate, input.channels);
        off.set_ratio(semis, tempo);
        off.set_transient_sched(false);
        report(
            &format!("{name} sched=off"),
            &off.render_to(&input.samples, target),
            &want,
            input.rate,
        );

        // Gates only (the shipped default): the state-2 chain gates without
        // hop re-timing.
        let mut on = VocoderState::new(input.rate, input.channels);
        on.set_ratio(semis, tempo);
        let got_on = on.render_to(&input.samples, target);
        let (ev, s2) = on.transient_stats();
        report(&format!("{name} sched=on "), &got_on, &want, input.rate);
        println!("  stats: {ev} event chunks, {s2} state-2 granules");

        // Experimental: hop re-timing on top of the gates. Opt-in — the
        // render doubles the suite time and the path is off by default
        // (`AU_AB_HOP=1 cargo test --release --test au_transient_ab ...`).
        if std::env::var("AU_AB_HOP").is_ok() {
            let mut hopt = VocoderState::new(input.rate, input.channels);
            hopt.set_ratio(semis, tempo);
            hopt.ts_hop_override = true;
            report(
                &format!("{name} sched=hop"),
                &hopt.render_to(&input.samples, target),
                &want,
                input.rate,
            );
        }

        // Experimental: the 13.8 cepstral gate (ts_rpt_b2) on top of the
        // default dual-queue path (`AU_AB_B2=1 ...`). The live probe saw the
        // DLL's own enable byte set during preview, so this measures the
        // gate's real contribution.
        if std::env::var("AU_AB_B2").is_ok() {
            let mut b2 = VocoderState::new(input.rate, input.channels);
            b2.set_ratio(semis, tempo);
            b2.cfg.ts_rpt_b2 = true;
            report(
                &format!("{name} sched=b2 "),
                &b2.render_to(&input.samples, target),
                &want,
                input.rate,
            );
        }
    }
    let _ = root();
}
