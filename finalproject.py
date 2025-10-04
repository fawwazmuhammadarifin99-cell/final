# Final Project!!
from __future__ import annotations

import os
import re
import requests
import streamlit as st
from openai import OpenAI
from typing import List, Dict, Tuple

# ===============================================================
# Setup: Secrets/Env & Konstanta
# ===============================================================
st.set_page_config(page_title="AI Dokter Remaja", page_icon="🩺", layout="centered")

# (Opsional untuk lokal saja) load .env jika modul tersedia
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

def get_secret(name: str, default: str | None = None) -> str:
    """Cloud: ambil dari st.secrets; Lokal: dari os.getenv (mis. .env)."""
    try:
        # st.secrets bisa berupa Mapping-like; gunakan get agar aman
        return st.secrets.get(name, os.getenv(name, default))  # type: ignore[attr-defined]
    except Exception:
        return os.getenv(name, default)

APP_TITLE = (
    "Prototype Pemanfaatan Kecerdasan Buatan (AI) sebagai Alat Bantu Diagnosis Masalah Kesehatan Murid SMP Labschool Jakarta"
)
APP_DESC = "Silakan jawab pertanyaan berikut untuk menganalisis masalah kesehatan anda."

# Kunci & model
OPENAI_API_KEY     = (get_secret("OPENAI_API_KEY", "") or "").strip()
OPENAI_MODEL       = (get_secret("OPENAI_MODEL", "gpt-4o-mini") or "").strip()

# SendGrid (Email)
SENDGRID_API_KEY   = (get_secret("SENDGRID_API_KEY", "") or "").strip()
EMAIL_FROM         = (get_secret("EMAIL_FROM", "") or "").strip()  # harus verified sender/domain di SendGrid

# Twilio (SMS) – opsional
TWILIO_ACCOUNT_SID = (get_secret("TWILIO_ACCOUNT_SID", "") or "").strip()
TWILIO_AUTH_TOKEN  = (get_secret("TWILIO_AUTH_TOKEN", "") or "").strip()
TWILIO_FROM        = (get_secret("TWILIO_FROM", "") or "").strip()

MISSING_SENDGRID = not (SENDGRID_API_KEY and EMAIL_FROM)

# Validasi minimum
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY belum disetel di Secrets/Environment. Set dulu di **Manage app → Settings → Secrets**.")
    st.stop()

# Client OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ===============================================================
# State Inisialisasi
# ===============================================================
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("chat_log", [])
    ss.setdefault("qa_pairs", [])
    ss.setdefault("step", "bio")           # 'bio' -> 'chat' -> 'done'
    ss.setdefault("bio_data", {})
    ss.setdefault("question_count", 0)
    ss.setdefault("max_questions", 10)
    ss.setdefault("final_analysis", None)
    ss.setdefault("first_question_sent", False)


# ===============================================================
# Utils & Parsers
# ===============================================================
def _clean_md(s: str) -> str:
    """Bersihkan markdown sederhana untuk tampilan plain text."""
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)          # **bold**
    s = re.sub(r"\*(.*?)\*", r"\1", s)              # *italic*
    s = re.sub(r"^#+\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*[-•]\s*", "- ", s, flags=re.MULTILINE)
    return s.strip()

def extract_selected_sections(full_text: str) -> str:
    """
    Ambil hanya:
      1) Kemungkinan diagnosis
      2) Rencana tindak lanjut & saran
      3) Edukasi pencegahan
    Toleran variasi heading.
    """
    text = full_text.strip()
    patterns = {
        "diagnosis": (
            r"(?:^|\n)\s*(?:\*\*)?\s*(?:kemungkinan\s*diagnosis|diagnosis(?:\s*diferensial)?)"
            r"\s*(?:\*\*)?\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:\*\*)?\s*"
            r"(?:rencana|saran|edukasi|pencegahan|kesimpulan|catatan)\b|$)"
        ),
        "plan": (
            r"(?:^|\n)\s*(?:\*\*)?\s*(?:rencana\s*tindak\s*lanjut(?:\s*&\s*saran)?|"
            r"rencana\s*tatalaksana|saran)\s*(?:\*\*)?\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:\*\*)?\s*"
            r"(?:edukasi|pencegahan|kemungkinan|diagnosis|kesimpulan|catatan)\b|$)"
        ),
        "edu": (
            r"(?:^|\n)\s*(?:\*\*)?\s*(?:edukasi\s*pencegahan|pencegahan)\s*(?:\*\*)?\s*[:\-]?\s*\n"
            r"(.*?)(?=\n\s*(?:\*\*)?\s*(?:rencana|saran|kemungkinan|diagnosis|kesimpulan|catatan)\b|$)"
        ),
    }

    out_parts = []
    for title, pat in [
        ("Kemungkinan Diagnosis", patterns["diagnosis"]),
        ("Rencana Tindak Lanjut & Saran", patterns["plan"]),
        ("Edukasi Pencegahan", patterns["edu"]),
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m and m.group(1).strip():
            content = _clean_md(m.group(1))
            out_parts.append(f"{title}:\n{content.strip()}")

    if not out_parts:
        fallback = _clean_md(text)
        return (fallback[:6000] + "…") if len(fallback) > 6000 else fallback

    final = "\n\n".join(out_parts).strip()
    return (final[:8000] + "…") if len(final) > 8000 else final

def normalize_msisdn(num: str) -> str:
    """Ubah 08xxxx menjadi +628xxxx; jaga jika sudah +..; hilangkan spasi/dash."""
    if not num:
        return ""
    s = re.sub(r"[^\d+]", "", num.strip())
    if s.startswith("+"):
        return s
    if s.startswith("0"):
        return "+62" + s[1:]
    if s.startswith("62"):
        return "+" + s
    if s.isdigit():
        return "+62" + s
    return s

def is_valid_email(addr: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", addr or ""))

def send_email_via_sendgrid(to_email: str, subject: str, html_body: str, text_body: str) -> None:
    if not SENDGRID_API_KEY or not EMAIL_FROM:
        raise RuntimeError("SENDGRID_API_KEY / EMAIL_FROM belum di-set.")
    if not to_email:
        raise RuntimeError("Alamat email tujuan kosong.")
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail

    message = Mail(
        from_email=EMAIL_FROM,
        to_emails=to_email,
        subject=subject,
        html_content=html_body,
        plain_text_content=text_body,
    )
    sg = SendGridAPIClient(SENDGRID_API_KEY)
    sg.send(message)

def send_sms_via_twilio(text_body: str, to_number: str) -> str:
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM):
        raise RuntimeError("Kredensial Twilio belum lengkap (TWILIO_ACCOUNT_SID/AUTH_TOKEN/TWILIO_FROM).")
    if not to_number:
        raise RuntimeError("Nomor tujuan SMS kosong.")
    from twilio.rest import Client as TwilioClient

    t_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    msg = t_client.messages.create(from_=TWILIO_FROM, to=to_number, body=text_body)
    return msg.sid

def _extract_diagnoses_from_analysis(analysis_text: str) -> List[str]:
    """Ambil daftar diagnosis dari blok 'Kemungkinan Diagnosis'."""
    pat = (
        r"(?:^|\n)\s*\*\*?\s*(?:kemungkinan\s*diagnosis|diagnosis(?:\s*diferensial)?)\s*\*\*?\s*[:\-]?\s*\n"
        r"(.*?)(?=\n\s*\*\*|\Z)"
    )
    m = re.search(pat, analysis_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []

    block = m.group(1)
    items = re.findall(r"(?:^\s*(?:[-•]|\d+\.)\s*)(.+)$", block, flags=re.MULTILINE)
    diagnoses: List[str] = []
    for it in items:
        it = re.sub(r"\s*\([^)]*\)", "", it)
        it = re.sub(r"[:\-–].*$", "", it).strip()
        diagnoses.append(it)
    if not diagnoses:
        diagnoses = [ln.strip() for ln in block.splitlines() if ln.strip()]

    seen, out = set(), []
    for d in diagnoses:
        key = d.lower()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def suggest_otc_plan(diagnoses: List[str], usia_tahun: int | str = 15, context_hint: str = "") -> Dict[str, object]:
    """Saran Obat OTC & Perawatan di Rumah (maks 7 poin)."""
    try:
        usia = int(usia_tahun)  # noqa: F841
    except Exception:
        usia = 15

    pool = " ".join(diagnoses or []).lower() + " " + (context_hint or "").lower()
    bullets: List[str] = []

    if any(k in pool for k in ["ruam", "kemerahan", "gatal", "biduran", "urtikaria", "dermatitis", "alergi kulit"]):
        bullets += [
            "Oles **hydrocortisone 1%** tipis 1–2×/hari (maks 7 hari; jangan pada luka/infeksi).",
            "**Cetirizine 10 mg** atau **loratadine** 1×/hari untuk gatal.",
            "**Lotion calamine** / **moisturizer hipoalergenik** secara rutin.",
            "Hindari pemicu (sabun keras, pewangi, makanan/paparan yang dicurigai).",
        ]
    if any(k in pool for k in ["batuk berdahak", "dahak", "lendir", "bronkitis", "mukus"]):
        bullets += [
            "**Guaifenesin** sesuai label (ekspektoran).",
            "Alternatif: **Bromhexine**/**Ambroxol**.",
            "**Semprot saline** hidung + inhalasi uap hangat 2–3×/hari.",
            "Madu 1 sdt sebelum tidur (khusus usia > 1 tahun).",
            "Dekongestan lokal **oksimetazolin 0,05%** sebelum tidur, **maks 3 hari**.",
        ]
    if any(k in pool for k in ["batuk kering", "non produktif"]):
        bullets += [
            "**Dextromethorphan** bila batuk mengganggu tidur.",
            "Hidrasi hangat, humidifier, dan permen pelega.",
        ]
    if any(k in pool for k in ["pilek", "hidung tersumbat", "flu", "rhinitis", "nasal congestion"]):
        bullets += [
            "**Semprot saline** rutin.",
            "**Xylometazoline 0,1%** atau **Oxymetazoline 0,05%** sebelum tidur, **maks 3 hari**.",
            "Jika alergi dominan: **cetirizine/loratadine** 1×/hari.",
        ]
    if any(k in pool for k in ["keseleo", "sprain", "strain", "terkilir", "tendinit"]):
        bullets += [
            "**RICE**: Rest, Ice 10–15 menit 3–4×/hari (48 jam pertama), Compression, Elevation.",
            "Oles **gel diklofenak 1%** 3–4×/hari.",
            "Gunakan penyangga sendi sementara; kembali ke aktivitas bertahap.",
        ]
    if any(k in pool for k in ["diare", "gastroenteritis", "mencret"]):
        bullets += [
            "**Oralit (ORS)** tiap BAB cair; minum sedikit tapi sering.",
            "**Zinc** 10–20 mg/hari selama 10–14 hari (bila tersedia).",
            "Hindari gorengan/pedas sementara; makan porsi kecil.",
        ]
    if any(k in pool for k in ["nyeri tenggorokan", "sakit tenggorokan", "faringitis", "radang tenggorokan"]):
        bullets += [
            "Kumur **air garam hangat** 3–4×/hari; pelega tenggorokan/lozenges.",
            "Spray kumur antiseptik (mis. **povidone-iodine**) sesuai label.",
        ]

    bullets = bullets[:7]
    safety = [
        "Selalu baca label & **ikuti dosis kemasan** (usia/berat).",
        "Hentikan bila muncul reaksi alergi/ruam hebat/bengkak/napas sesak.",
        "Ke IGD jika **red flag**: sesak berat, demam ≥39°C >3 hari, muntah terus, lemas/pingsan, nyeri hebat memburuk, perdarahan, kaku kuduk.",
    ]

    def _md(title: str, bl: List[str], sf: List[str]) -> str:
        return "### " + title + "\n" + "\n".join(f"- {b}" for b in bl) + "\n\n" + "\n".join(f"- {s}" for s in sf)

    def _html(title: str, bl: List[str], sf: List[str]) -> str:
        li_bullets = "".join(f"<li>{i}</li>" for i in bl)
        li_safety = "".join(f"<li>{i}</li>" for i in sf)
        return (
            f"<h3 style='margin:0 0 8px'>{title}</h3>"
            f"<ul style='margin:0 8px 8px 20px'>{li_bullets}</ul>"
            f"<ul style='color:#444;margin:0 0 0 20px'>{li_safety}</ul>"
        )

    title = "Saran Obat OTC & Perawatan di Rumah"
    return {"title": title, "bullets": bullets, "safety": safety, "md": _md(title, bullets, safety), "html": _html(title, bullets, safety)}


# ===============================================================
# Model Helpers (OpenAI)
# ===============================================================
def generate_next_question(qa_pairs: List[Tuple[str, str]]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah Dokter Spesialis, lulusan FK UI & S2 Johns Hopkins. "
                "Lakukan anamnesis MEDIS TERSTRUKTUR, berbasis bukti, fokus penyakit umum tropis. "
                "Pertanyaan lanjutan wajib berdasar jawaban terakhir dan relevansi klinis. "
                "Boleh cek red flag (sesak berat, nyeri dada hebat, kejang, penurunan kesadaran, bibir/kuku membiru, perdarahan hebat) dengan pertanyaan spesifik. "
                "KELUARAN: hanya SATU kalimat tanya paling diagnostik."
            ),
        },
        {
            "role": "user",
            "content": "Berikut riwayat percakapan:\n" + "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs]),
        },
    ]
    resp = openai_client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return (resp.choices[0].message.content or "").strip()

def fetch_research_summary() -> str:
    """Ambil cuplikan dari sumber resmi (best-effort, offline fallback aman)."""
    try:
        sources = [
            "https://www.who.int/health-topics/dengue-and-severe-dengue",
            "https://www.cdc.gov/dengue/index.html",
            "https://www.cdc.gov/malaria/index.html",
            "https://www.idai.or.id/",
            "https://www.kemkes.go.id/",
        ]
        summary = ""
        for url in sources:
            resp = requests.get(url, timeout=10)
            if resp.ok:
                text_snippet = resp.text[:1500]
                summary += f"Sumber: {url}\nCuplikan: {text_snippet}\n\n"
        return summary or "Tidak ada ringkasan yang dapat diambil saat ini."
    except Exception:
        return "Tidak dapat mengambil ringkasan riset saat ini."

def analyze_health(bio: Dict[str, str], qa_pairs: List[Tuple[str, str]], research_summary: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah Dokter Spesialis lulusan FK UI dan S2 Johns Hopkins. "
                "Lakukan analisis berbasis bukti dan buat diagnosis diferensial dari anamnesis. "
                "Susun output dengan heading tebal: "
                "(1) Ringkasan Gejala, (2) Kemungkinan Diagnosis (dengan alasan), "
                "(3) Rencana Tindak Lanjut & Saran (spesifik), (4) Edukasi Pencegahan. "
                "Hindari kepastian absolut; tandai red flag bila ada."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Biodata:\n"
                f"Nama: {bio.get('nama','-')}\n"
                f"Usia: {bio.get('usia','-')}\n"
                f"Kelas: {bio.get('kelas','-')}\n"
                f"Jenis Kelamin: {bio.get('jenis_kelamin','-')}\n\n"
                f"Percakapan Q/A:\n" + "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs]) +
                f"\n\nRingkasan riset (opsional):\n{research_summary}\n"
                "\nBerikan analisis sesuai format."
            ),
        },
    ]
    resp = openai_client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return (resp.choices[0].message.content or "").strip()


# ===============================================================
# UI
# ===============================================================
def _render_header() -> None:
    st.markdown(f"<h1 style='text-align: center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{APP_DESC}</p>", unsafe_allow_html=True)
    st.divider()

def _bio_form() -> None:
    """Form biodata yang menghilang setelah tombol 'Lanjut' ditekan."""
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("**Isi biodata singkat:**")
        with st.form("bio_form", clear_on_submit=False):
            nama = st.text_input("Nama (opsional)",
                                 value=st.session_state.bio_data.get("nama", ""))
            try:
                usia_default = int(st.session_state.bio_data.get("usia", 13))
            except Exception:
                usia_default = 13
            usia = st.number_input("Usia (tahun)", min_value=7, max_value=20, step=1, value=usia_default)

            kelas_current = str(st.session_state.bio_data.get("kelas", "7"))
            kelas = st.selectbox("Kelas", options=["7", "8", "9"],
                                 index=["7", "8", "9"].index(kelas_current) if kelas_current in ["7", "8", "9"] else 0)

            jk = st.selectbox("Jenis Kelamin", options=["Laki-Laki", "Perempuan"],
                              index=0 if st.session_state.bio_data.get("jenis_kelamin", "L") == "L" else 1)

            email = st.text_input("Email (untuk menerima hasil lengkap via email)",
                                  placeholder="nama@sekolah.sch.id",
                                  value=st.session_state.bio_data.get("email", ""))

            nohp = st.text_input("Nomor HP (untuk menerima SMS notifikasi)",
                                 placeholder="+62812xxxxxxx",
                                 value=st.session_state.bio_data.get("nohp", ""))

            submit = st.form_submit_button("Lanjut")

    if submit:
        norm = normalize_msisdn(nohp) if nohp else ""
        st.session_state.bio_data.update({
            "nama": nama.strip(),
            "usia": str(usia),
            "kelas": str(kelas),
            "jenis_kelamin": jk,
            "email": email.strip(),
            "nohp": norm or nohp,
        })
        if email and not is_valid_email(email):
            st.warning("Format email kurang tepat. Anda tetap bisa lanjut, tetapi pengiriman email mungkin gagal.")

        st.session_state.step = "chat"
        if not st.session_state.first_question_sent:
            q = "Bisa diceritakan dengan lengkap, Anda saat ini mengalami keluhan kesehatan apa?"
            st.session_state.chat_log.append({"role": "assistant", "content": q})
            st.session_state.first_question_sent = True

        placeholder.empty()
        st.rerun()

def _render_chat_history() -> None:
    for msg in st.session_state.chat_log:
        role = "assistant" if msg["role"] == "assistant" else "user"
        st.chat_message(role).markdown(msg["content"])

def _handle_chat_flow() -> None:
    user_input = st.chat_input("Jawaban Anda...") if st.session_state.step in ("chat", "done") else None
    if not (user_input and st.session_state.step == "chat"):
        return

    # Tampilkan & simpan jawaban user
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_log.append({"role": "user", "content": user_input})

    # Ambil pertanyaan asisten terakhir (sebelum jawaban user)
    last_q = next(
        (m["content"] for m in reversed(st.session_state.chat_log[:-1]) if m["role"] == "assistant"),
        "(pertanyaan awal)",
    )
    st.session_state.qa_pairs.append((last_q, user_input))
    st.session_state.question_count += 1

    # Jika sudah mencapai batas pertanyaan, lakukan analisis final
    if st.session_state.question_count >= st.session_state.max_questions:
        with st.spinner("Mengambil referensi riset..."):
            research_summary = fetch_research_summary()

        with st.spinner("Menganalisis jawaban Anda berdasarkan riset..."):
            result = analyze_health(st.session_state.bio_data, st.session_state.qa_pairs, research_summary)

        # ======== HASIL BAGIAN 1: Analisis ========
        st.session_state.final_analysis = result
        st.chat_message("assistant").markdown("*Hasil analisis masalah kesehatan Anda:*")
        st.chat_message("assistant").markdown(result)
        st.session_state.chat_log.append({"role": "assistant", "content": result})
        st.session_state.step = "done"

        # ======== HASIL BAGIAN 2: Saran Obat OTC ========
        diag_list = _extract_diagnoses_from_analysis(result)
        last_answer = st.session_state.qa_pairs[-1][1] if st.session_state.qa_pairs else ""
        otc_plan = suggest_otc_plan(diag_list, st.session_state.bio_data.get("usia", "15"), context_hint=(result + " " + last_answer))

        st.chat_message("assistant").markdown(otc_plan["md"])
        st.session_state.chat_log.append({"role": "assistant", "content": otc_plan["md"]})

        # ====== Siapkan konten email ======
        nama = st.session_state.bio_data.get("nama", "Siswa")
        selected = extract_selected_sections(result)
        subject = f"Hasil AI Dokter Remaja — {nama}"

        text_body = (
            f"Halo {nama},\n\n"
            "Berikut hasil analisis AI Dokter Remaja:\n\n"
            f"{selected}\n\n"
            "-----\n"
            "Saran Obat OTC & Perawatan di Rumah\n"
            + "\n".join(f"- {b}" for b in otc_plan["bullets"]) + "\n\n"
            + "\n".join(f"- {s}" for s in otc_plan["safety"]) + "\n\n"
            "Disclaimer: Ini bukan diagnosis resmi. Jika Anda mengalami tanda bahaya atau nyeri berat, segera cari pertolongan medis darurat."
        )

        html_body = (
            f"<p>Halo <b>{nama}</b>,</p>"
            f"<p>Berikut hasil analisis <b>AI Dokter Remaja</b>:</p>"
            f"<div style='border:1px solid #eee;padding:12px;border-radius:8px;white-space:pre-wrap;font-family:system-ui,Segoe UI,Arial;'>"
            f"{selected}</div>"
            f"<div style='height:10px'></div>"
            f"<div style='border:1px solid #eee;padding:12px;border-radius:8px;font-family:system-ui,Segoe UI,Arial;'>"
            f"{otc_plan['html']}</div>"
            "<p style='color:#666'><i>Disclaimer: Ini bukan diagnosis resmi. "
            "Jika Anda mengalami tanda bahaya atau nyeri berat, segera cari pertolongan medis darurat.</i></p>"
        )

        # ====== Kirim Email via SendGrid ======
        to_email = (st.session_state.bio_data.get("email") or "").strip()
        if to_email:
            if MISSING_SENDGRID:
                st.warning("Email tidak terkirim: kredensial SendGrid belum disetel (SENDGRID_API_KEY/EMAIL_FROM).")
            else:
                try:
                    send_email_via_sendgrid(to_email, subject, html_body, text_body)
                    st.success(f"Email terkirim ke {to_email}")
                except Exception as e:
                    st.warning(f"Email tidak terkirim: {e}")
        else:
            st.info("Email tidak diisi, jadi hasil lengkap tidak dikirim via email.")

        # ====== Kirim SMS notifikasi singkat (opsional) ======
        to_num = normalize_msisdn(st.session_state.bio_data.get("nohp", ""))
        if to_num:
            if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM:
                sms_text = (
                    f"Halo {nama}, hasil AI Dokter Remaja sudah dikirim ke email Anda. "
                    f"Silakan cek inbox/SPAM dengan subjek: '{subject}'."
                )
                try:
                    sid = send_sms_via_twilio(sms_text, to_num)
                    st.success(f"SMS notifikasi terkirim ke {to_num}. SID: {sid}")
                except Exception as e:
                    st.warning(f"SMS tidak terkirim: {e}")
            else:
                st.info("Kredensial Twilio belum lengkap, SMS tidak dikirim.")
        else:
            st.info("Nomor HP tidak diisi, jadi tidak ada SMS notifikasi yang dikirim.")
        return

    # Lanjutkan anamnesis
    with st.spinner("Mempersiapkan pertanyaan selanjutnya..."):
        next_q = generate_next_question(st.session_state.qa_pairs)
    st.chat_message("assistant").markdown(next_q)
    st.session_state.chat_log.append({"role": "assistant", "content": next_q})


def main() -> None:
    _init_state()
    _render_header()

    # UI Biodata (sekali tampil – hilang setelah 'Lanjut')
    if st.session_state.step == "bio":
        _bio_form()

    # Tampilkan chat yang sudah ada
    _render_chat_history()

    # Alur chat
    _handle_chat_flow()

    # Disclaimer global
    st.info(
        "Disclaimer: Ini bukan diagnosis resmi. Jika Anda mengalami tanda bahaya atau nyeri berat, "
        "segera cari pertolongan medis darurat."
    )


if __name__ == "__main__":
    main()
