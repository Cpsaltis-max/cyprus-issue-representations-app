import streamlit as st
from supabase import create_client
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import pyreadstat
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Cyprus Issue Observatory", layout="wide")

# ============================================================
# LOGOS
# ============================================================
GSP_LOGO = Path("gsp_logo.png")
UCFS_LOGO = Path("ucfs_logo.png")
HISTORICAL_DATA_CANDIDATES = [
    Path("data/historical_responses.sav"),
    Path("data/historical_responses.csv"),
    Path("historical_responses.sav"),
    Path("historical_responses.csv"),
]

def show_logo_header():
    """Show GSP and UCFS logos if the PNG files are present in the app folder."""
    if not GSP_LOGO.exists() and not UCFS_LOGO.exists():
        return
    left, middle, right = st.columns([1.1, 3.5, 1.1])
    with left:
        if GSP_LOGO.exists():
            st.image(str(GSP_LOGO), use_container_width=True)
    with right:
        if UCFS_LOGO.exists():
            st.image(str(UCFS_LOGO), use_container_width=True)
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

def get_supabase_client():
    try:
        supabase_url = str(st.secrets["SUPABASE_URL"]).strip().rstrip("/")
        supabase_key = str(st.secrets["SUPABASE_KEY"]).strip()
    except KeyError:
        st.error(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to the app secrets in Streamlit Cloud."
        )
        st.stop()

    if not supabase_url.startswith(("https://", "http://")):
        st.error(
            "SUPABASE_URL must be the Supabase project API URL, for example "
            "https://your-project-ref.supabase.co."
        )
        st.stop()

    if not supabase_key:
        st.error("SUPABASE_KEY is empty. Add your Supabase anon key to Streamlit secrets.")
        st.stop()

    return create_client(supabase_url, supabase_key)

supabase = get_supabase_client()

TOTAL_PAGES = 12

# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "data" not in st.session_state:
    st.session_state.data = {}

if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now(timezone.utc)

# ============================================================
# LANGUAGE
# ============================================================

lang_options = {
    "en": "English",
    "el": "Ελληνικά",
    "tr": "Türkçe"
}

lang = st.sidebar.selectbox(
    "Language / Γλώσσα / Dil",
    list(lang_options.keys()),
    format_func=lambda x: lang_options[x],
    key="language_selector"
)

# ============================================================
# TRANSLATIONS
# ============================================================

T = {
    "en": {
        "title": "Cyprus Issue Observatory",
        "caption": "Research instrument — University of Cyprus",
        "next": "Next",
        "back": "Back",
        "submit": "Submit",
        "page": "Page",
        "success": "Submitted successfully.",
        "saved_response": "Saved response:",

        "community": "Community",
        "gc": "Greek Cypriot",
        "tc": "Turkish Cypriot",
        "age": "Age",
        "yearborn": "Year born",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "education": "Education",
        "area": "Area",
        "urban": "Urban",
        "rural": "Rural",
        "refusal": "Refusal",
        "dk": "Don't know",
        "nr": "NR",
        "dk_nr": "DK / NR",

        "reads_writes": "Reads and writes",
        "elementary": "Elementary",
        "gymnasium": "Gymnasium",
        "lyceum": "Lyceum",
        "college": "College",
        "university": "University",
        "postgrad": "Postgraduate studies",
        "na_dk": "NA / DK",

        "yes": "Yes",
        "no": "No",

        "p1": "1. Demographics",
        "p2": "2. Identity Orientation",
        "p3": "3. Religiosity",
        "p4": "4. Displacement and Property",
        "p5": "5. Contact Frequency",
        "p6": "6. Quality of Contact",
        "p7": "7. Cohabitation",
        "p8": "8. Feelings",
        "p9": "9. Trust",
        "p10": "10. Threats",
        "p11": "11. Community Identity",
        "p12": "12. Views on Possible Solutions",

        "moreno_stem": "Which of the following best describes how you feel?",
        "only_cypriot": "Only Cypriot and not at all {motherland}",
        "cypriot_bit": "Cypriot and a bit {motherland}",
        "equal_cy_mother": "To the same extent Cypriot and {motherland}",
        "mother_bit_cy": "{motherland} and a bit Cypriot",
        "only_mother": "Only {motherland} and not at all Cypriot",

        "religion_important": "How important is religion to you?",
        "religion_practice": "Apart from special occasions such as weddings, funerals, baptisms and so on, how often nowadays do you attend services or prayer connected with your religion?",
        "not_at_all": "Not at all",
        "little_importance": "Of little importance",
        "important": "It is important",
        "very_important": "Very important",
        "extremely_important": "Extremely important",
        "never_practically": "Never or practically never",
        "once_year": "At least once a year",
        "once_month": "At least once a month",
        "once_two_weeks": "At least once every two weeks",
        "once_week": "At least once a week",

        "idp_self": "Were you personally displaced / made a refugee from the {place} because of the events of {events} in Cyprus?",
        "idp_family": "Did one or more of your parents or grandparents have to leave their place of residence in the {place} because of the events of {events} in Cyprus?",
        "property": "Do you own property in the {place}?",

        "contact_stem": "Thinking about your social contacts — communication or conversations — how often do you have contact these days with {other} in the following situations?",
        "never": "Never",
        "less_month": "Less than once a month",
        "once_month_contact": "Once a month",
        "several_month": "Several times a month",
        "once_week_contact": "Once a week",
        "several_week": "Several times a week",
        "every_day": "Every day",
        "work": "At work",
        "bicommunal": "At bicommunal events",
        "neighbourhood": "In your neighbourhood",
        "occupied_gc": "In the occupied areas",
        "occupied_tc": "In the non-government controlled areas / north",
        "nonoccupied_gc": "In the non-occupied areas in general",
        "nonoccupied_tc": "In the government-controlled areas / south in general",
        "social_media": "On social media, e.g. Facebook, Instagram",

        "quality_stem": "When you meet with {other} anywhere in Cyprus, in general how do you find the contact?",
        "little_bit": "A little bit",
        "moderately": "Moderately",
        "quite_bit": "Quite a bit",
        "very_much": "Very much",
        "pleasant": "Pleasant",
        "superficial": "Superficial",
        "coop": "In cooperative spirit",
        "positive": "Positive",
        "mutual_respect": "Based on mutual respect",

        "cohab_stem": "Please indicate whether you agree or disagree with the following statements.",
        "abs_disagree": "Absolutely disagree",
        "disagree": "Disagree",
        "neither": "Neither agree nor disagree",
        "agree": "Agree",
        "abs_agree": "Absolutely agree",
        "live_together": "I feel that I can live together with {other}",
        "neighbors": "I would not mind having {other} as neighbors",

        "thermo_stem": "The following question concerns your feelings toward {other} in general. Please rate this group of people on a scale from 0 to 10. The higher the score, the warmer or more positive you feel. The lower the score, the colder or more negative you feel. If you feel neither warm nor cold, choose 5.",
        "thermo_q": "How do you feel toward {other} in general?",

        "trust_stem": "Now we would like to ask some questions about {other} in general. Please indicate whether you agree or disagree with the following statements.",
        "trust_love": "I trust {other} when they say that they love Cyprus",
        "no_trust_politicians": "I find it hard to trust {other_singular} politicians when it comes to the implementation of an agreed solution",
        "trust_ordinary": "I trust {other_singular} ordinary people when they say they want peace",
        "trust_politicians": "I trust {other_singular} politicians when they say they want peace",

        "threat_stem": "To what extent do you agree or disagree with the following statements?",
        "rthreat_power": "The more power {other} gain in this country, the more difficult it is for {own}",
        "rthreat_political": "Allowing {other} to decide on political issues means that {own} have less to say in how this country is run",
        "rthreat_claim_more": "I worry that {other} will claim more and more from us in the future",
        "sthreat_values": "Turkish Cypriots and Greek Cypriots in Cyprus have very different values",
        "sthreat_do_things": "{other} sometimes do things that {own} would never do",

        "identity_stem": "Some statements follow. Please let us know the degree of your agreement or disagreement with each statement.",
        "id_happy": "In general, I am happy to be a {own_singular}",
        "id_proud": "I am proud to be a {own_singular}",
        "id_important": "The fact that I am a {own_singular} is an important part of my identity",
        "id_self": "Being a {own_singular} is an important part of how I see myself",

        "solution_stem": "Which of the following three options — against, in favour, or neither against nor in favour but you could tolerate it if necessary — would you choose for each of the following possible solutions to the Cyprus problem?",
        "against": "Against",
        "tolerate": "Neither against nor in favour, but I could tolerate it if necessary",
        "in_favour": "In favour",
        "status_quo": "Keep the status quo",
        "bbf": "Bizonal, bicommunal federation",
        "unitary": "Unitary state",
        "two_states": "Two-state solution"
    },

    "el": {
        "title": "Παρατηρητήριο Κυπριακού Ζητήματος",
        "caption": "Ερευνητικό εργαλείο — Πανεπιστήμιο Κύπρου",
        "next": "Επόμενο",
        "back": "Πίσω",
        "submit": "Υποβολή",
        "page": "Σελίδα",
        "success": "Η απάντηση υποβλήθηκε με επιτυχία.",
        "saved_response": "Αποθηκευμένη απάντηση:",

        "community": "Κοινότητα",
        "gc": "Ελληνοκύπριος/α",
        "tc": "Τουρκοκύπριος/α",
        "age": "Ηλικία",
        "yearborn": "Έτος γέννησης",
        "gender": "Φύλο",
        "male": "Άνδρας",
        "female": "Γυναίκα",
        "education": "Εκπαίδευση",
        "area": "Περιοχή",
        "urban": "Αστική",
        "rural": "Αγροτική",
        "refusal": "Άρνηση",
        "dk": "Δεν γνωρίζω",
        "nr": "ΔΑ",
        "dk_nr": "Δεν γνωρίζω / Δεν απαντώ",

        "reads_writes": "Διαβάζει και γράφει",
        "elementary": "Δημοτικό",
        "gymnasium": "Γυμνάσιο",
        "lyceum": "Λύκειο",
        "college": "Κολλέγιο",
        "university": "Πανεπιστήμιο",
        "postgrad": "Μεταπτυχιακές σπουδές",
        "na_dk": "Δεν απαντώ / Δεν γνωρίζω",

        "yes": "Ναι",
        "no": "Όχι",

        "p1": "1. Δημογραφικά",
        "p2": "2. Προσανατολισμός ταυτότητας",
        "p3": "3. Θρησκευτικότητα",
        "p4": "4. Εκτοπισμός και περιουσία",
        "p5": "5. Συχνότητα επαφής",
        "p6": "6. Ποιότητα επαφής",
        "p7": "7. Συμβίωση",
        "p8": "8. Συναισθήματα",
        "p9": "9. Εμπιστοσύνη",
        "p10": "10. Απειλές",
        "p11": "11. Κοινοτική ταυτότητα",
        "p12": "12. Απόψεις για πιθανές λύσεις",

        "moreno_stem": "Ποιο από τα ακόλουθα περιγράφει καλύτερα το πώς αισθάνεστε;",
        "only_cypriot": "Μόνο Κύπριος/α και καθόλου {motherland}",
        "cypriot_bit": "Κύπριος/α και λίγο {motherland}",
        "equal_cy_mother": "Στον ίδιο βαθμό Κύπριος/α και {motherland}",
        "mother_bit_cy": "{motherland} και λίγο Κύπριος/α",
        "only_mother": "Μόνο {motherland} και καθόλου Κύπριος/α",

        "religion_important": "Πόσο σημαντική είναι η θρησκεία για εσάς;",
        "religion_practice": "Εκτός από ειδικές περιστάσεις όπως γάμοι, κηδείες, βαφτίσεις κ.λπ., πόσο συχνά σήμερα παρακολουθείτε λειτουργίες ή προσευχές που συνδέονται με τη θρησκεία σας;",
        "not_at_all": "Καθόλου",
        "little_importance": "Λίγο σημαντική",
        "important": "Σημαντική",
        "very_important": "Πολύ σημαντική",
        "extremely_important": "Εξαιρετικά σημαντική",
        "never_practically": "Ποτέ ή σχεδόν ποτέ",
        "once_year": "Τουλάχιστον μία φορά τον χρόνο",
        "once_month": "Τουλάχιστον μία φορά τον μήνα",
        "once_two_weeks": "Τουλάχιστον μία φορά κάθε δύο εβδομάδες",
        "once_week": "Τουλάχιστον μία φορά την εβδομάδα",

        "idp_self": "Εκτοπιστήκατε / γίνατε πρόσφυγας από την {place} εξαιτίας των γεγονότων του {events} στην Κύπρο;",
        "idp_family": "Ένας ή περισσότεροι από τους γονείς ή παππούδες/γιαγιάδες σας αναγκάστηκαν να εγκαταλείψουν τον τόπο διαμονής τους στην {place} εξαιτίας των γεγονότων του {events} στην Κύπρο;",
        "property": "Έχετε περιουσία στην {place};",

        "contact_stem": "Σκεπτόμενοι τις κοινωνικές σας επαφές — επικοινωνία ή συζητήσεις — πόσο συχνά έχετε σήμερα επαφή με {other} στις ακόλουθες περιπτώσεις;",
        "never": "Ποτέ",
        "less_month": "Λιγότερο από μία φορά τον μήνα",
        "once_month_contact": "Μία φορά τον μήνα",
        "several_month": "Αρκετές φορές τον μήνα",
        "once_week_contact": "Μία φορά την εβδομάδα",
        "several_week": "Αρκετές φορές την εβδομάδα",
        "every_day": "Κάθε μέρα",
        "work": "Στην εργασία",
        "bicommunal": "Σε δικοινοτικές εκδηλώσεις",
        "neighbourhood": "Στη γειτονιά σας",
        "occupied_gc": "Στις κατεχόμενες περιοχές",
        "occupied_tc": "Στο Βορρά",
        "nonoccupied_gc": "Στις μη κατεχόμενες περιοχές",
        "nonoccupied_tc": "Στο Νότο",
        "social_media": "Στα μέσα κοινωνικής δικτύωσης, π.χ. Facebook, Instagram",

        "quality_stem": "Όταν συναντάτε {other} οπουδήποτε στην Κύπρο, γενικά πώς βρίσκετε την επαφή;",
        "little_bit": "Λίγο",
        "moderately": "Μέτρια",
        "quite_bit": "Αρκετά",
        "very_much": "Πάρα πολύ",
        "pleasant": "Ευχάριστη",
        "superficial": "Επιφανειακή",
        "coop": "Με πνεύμα συνεργασίας",
        "positive": "Θετική",
        "mutual_respect": "Βασισμένη στον αμοιβαίο σεβασμό",

        "cohab_stem": "Παρακαλούμε δηλώστε κατά πόσο συμφωνείτε ή διαφωνείτε με τις ακόλουθες δηλώσεις.",
        "abs_disagree": "Διαφωνώ απόλυτα",
        "disagree": "Διαφωνώ",
        "neither": "Ούτε συμφωνώ ούτε διαφωνώ",
        "agree": "Συμφωνώ",
        "abs_agree": "Συμφωνώ απόλυτα",
        "live_together": "Αισθάνομαι ότι μπορώ να ζήσω μαζί με {other}",
        "neighbors": "Δεν θα με πείραζε να έχω {other} ως γείτονες",

        "thermo_stem": "Η ακόλουθη ερώτηση αφορά τα συναισθήματά σας προς {other} γενικά. Παρακαλούμε βαθμολογήστε αυτή την ομάδα από 0 έως 10. Όσο μεγαλύτερη είναι η βαθμολογία, τόσο πιο θερμά ή θετικά αισθάνεστε. Όσο χαμηλότερη είναι η βαθμολογία, τόσο πιο ψυχρά ή αρνητικά αισθάνεστε. Αν δεν αισθάνεστε ούτε θερμά ούτε ψυχρά, επιλέξτε 5.",
        "thermo_q": "Πώς αισθάνεστε γενικά προς {other};",

        "trust_stem": "Τώρα θα θέλαμε να σας κάνουμε μερικές ερωτήσεις για {other} γενικά. Παρακαλούμε δηλώστε κατά πόσο συμφωνείτε ή διαφωνείτε με τις ακόλουθες δηλώσεις.",
        "trust_love": "Εμπιστεύομαι {other} όταν λένε ότι αγαπούν την Κύπρο",
        "no_trust_politicians": "Δυσκολεύομαι να εμπιστευτώ {other_singular} πολιτικούς όσον αφορά την εφαρμογή μιας συμφωνημένης λύσης",
        "trust_ordinary": "Εμπιστεύομαι απλούς/ές {other_singular} όταν λένε ότι θέλουν ειρήνη",
        "trust_politicians": "Εμπιστεύομαι {other_singular} πολιτικούς όταν λένε ότι θέλουν ειρήνη",

        "threat_stem": "Σε ποιο βαθμό συμφωνείτε ή διαφωνείτε με τις ακόλουθες δηλώσεις;",
        "rthreat_power": "Όσο περισσότερη δύναμη αποκτούν {other}, τόσο πιο δύσκολα γίνονται τα πράγματα για {own}",
        "rthreat_political": "Το να επιτρέπεται σε {other} να αποφασίζουν για πολιτικά ζητήματα σημαίνει ότι {own} έχουν λιγότερο λόγο στον τρόπο που κυβερνάται αυτή η χώρα",
        "rthreat_claim_more": "Ανησυχώ ότι {other} θα διεκδικούν όλο και περισσότερα από εμάς στο μέλλον",
        "sthreat_values": "Οι Τουρκοκύπριοι και οι Ελληνοκύπριοι στην Κύπρο έχουν πολύ διαφορετικές αξίες",
        "sthreat_do_things": "{other} μερικές φορές κάνουν πράγματα που {own} δεν θα έκαναν ποτέ",

        "identity_stem": "Ακολουθούν ορισμένες δηλώσεις. Παρακαλούμε δηλώστε τον βαθμό συμφωνίας ή διαφωνίας σας με καθεμία.",
        "id_happy": "Γενικά, είμαι χαρούμενος/η που είμαι {own_singular}",
        "id_proud": "Είμαι περήφανος/η που είμαι {own_singular}",
        "id_important": "Το γεγονός ότι είμαι {own_singular} είναι σημαντικό μέρος της ταυτότητάς μου",
        "id_self": "Το να είμαι {own_singular} είναι σημαντικό μέρος του πώς βλέπω τον εαυτό μου",

        "solution_stem": "Ποια από τις ακόλουθες τρεις επιλογές — κατά, υπέρ ή ούτε κατά ούτε υπέρ αλλά θα μπορούσατε να την ανεχθείτε αν ήταν αναγκαίο — θα επιλέγατε για καθεμία από τις ακόλουθες πιθανές λύσεις του Κυπριακού;",
        "against": "Κατά",
        "tolerate": "Ούτε κατά ούτε υπέρ, αλλά θα μπορούσα να το ανεχθώ αν ήταν αναγκαίο",
        "in_favour": "Υπέρ",
        "status_quo": "Διατήρηση της υφιστάμενης κατάστασης",
        "bbf": "Διζωνική Δικοινοτική Ομοσπονδία",
        "unitary": "Ενιαίο κράτος",
        "two_states": "Λύση δύο κρατών"
    },

    "tr": {
        "title": "Kıbrıs Sorunu Gözlemevi",
        "caption": "Araştırma aracı — Kıbrıs Üniversitesi",
        "next": "İleri",
        "back": "Geri",
        "submit": "Gönder",
        "page": "Sayfa",
        "success": "Yanıt başarıyla kaydedildi.",
        "saved_response": "Kaydedilen yanıt:",

        "community": "Toplum",
        "gc": "Kıbrıslı Rum",
        "tc": "Kıbrıslı Türk",
        "age": "Yaş",
        "yearborn": "Doğum yılı",
        "gender": "Cinsiyet",
        "male": "Erkek",
        "female": "Kadın",
        "education": "Eğitim",
        "area": "Yerleşim yeri",
        "urban": "Kentsel",
        "rural": "Kırsal",
        "refusal": "Reddetti",
        "dk": "Bilmiyorum",
        "nr": "Cevap yok",
        "dk_nr": "Bilmiyorum / Cevap yok",

        "reads_writes": "Okur yazar",
        "elementary": "İlkokul",
        "gymnasium": "Ortaokul",
        "lyceum": "Lise",
        "college": "Kolej",
        "university": "Üniversite",
        "postgrad": "Lisansüstü eğitim",
        "na_dk": "Cevap yok / Bilmiyorum",

        "yes": "Evet",
        "no": "Hayır",

        "p1": "1. Demografi",
        "p2": "2. Kimlik yönelimi",
        "p3": "3. Dindarlık",
        "p4": "4. Yerinden edilme ve mülkiyet",
        "p5": "5. Temas sıklığı",
        "p6": "6. Temas kalitesi",
        "p7": "7. Birlikte yaşama",
        "p8": "8. Duygular",
        "p9": "9. Güven",
        "p10": "10. Tehditler",
        "p11": "11. Toplumsal kimlik",
        "p12": "12. Olası çözümler hakkındaki görüşler",

        "moreno_stem": "Aşağıdakilerden hangisi kendinizi nasıl hissettiğinizi en iyi açıklar?",
        "only_cypriot": "Sadece Kıbrıslı ve hiç {motherland} değil",
        "cypriot_bit": "Kıbrıslı ve biraz {motherland}",
        "equal_cy_mother": "Aynı derecede Kıbrıslı ve {motherland}",
        "mother_bit_cy": "{motherland} ve biraz Kıbrıslı",
        "only_mother": "Sadece {motherland} ve hiç Kıbrıslı değil",

        "religion_important": "Din sizin için ne kadar önemlidir?",
        "religion_practice": "Düğün, cenaze, vaftiz gibi özel durumlar dışında, bugünlerde dininizle bağlantılı ibadet veya dualara ne sıklıkla katılırsınız?",
        "not_at_all": "Hiç",
        "little_importance": "Az önemli",
        "important": "Önemli",
        "very_important": "Çok önemli",
        "extremely_important": "Son derece önemli",
        "never_practically": "Hiç ya da neredeyse hiç",
        "once_year": "Yılda en az bir kez",
        "once_month": "Ayda en az bir kez",
        "once_two_weeks": "İki haftada en az bir kez",
        "once_week": "Haftada en az bir kez",

        "idp_self": "Kıbrıs'ta {events} olayları nedeniyle {place} bölgesinden kişisel olarak yerinden edildiniz / mülteci oldunuz mu?",
        "idp_family": "Anne-babanızdan veya büyük anne-büyük babalarınızdan biri ya da daha fazlası Kıbrıs'ta {events} olayları nedeniyle {place} bölgesindeki ikamet yerini terk etmek zorunda kaldı mı?",
        "property": "{place} bölgesinde mülkünüz var mı?",

        "contact_stem": "Sosyal temaslarınızı — iletişim veya konuşmalarınızı — düşünerek, bugünlerde aşağıdaki durumlarda {other} ile ne sıklıkla temasınız oluyor?",
        "never": "Hiç",
        "less_month": "Ayda birden az",
        "once_month_contact": "Ayda bir",
        "several_month": "Ayda birkaç kez",
        "once_week_contact": "Haftada bir",
        "several_week": "Haftada birkaç kez",
        "every_day": "Her gün",
        "work": "İş yerinde",
        "bicommunal": "İki toplumlu etkinliklerde",
        "neighbourhood": "Mahallenizde",
        "occupied_gc": "İşgal altındaki bölgelerde",
        "occupied_tc": "Hükümet kontrolünde olmayan bölgelerde / kuzeyde",
        "nonoccupied_gc": "İşgal altında olmayan bölgelerde genel olarak",
        "nonoccupied_tc": "Hükümet kontrolündeki bölgelerde / güneyde genel olarak",
        "social_media": "Sosyal medyada, örn. Facebook, Instagram",

        "quality_stem": "Kıbrıs'ın herhangi bir yerinde {other} ile karşılaştığınızda, genel olarak bu teması nasıl buluyorsunuz?",
        "little_bit": "Biraz",
        "moderately": "Orta derecede",
        "quite_bit": "Oldukça",
        "very_much": "Çok fazla",
        "pleasant": "Hoş",
        "superficial": "Yüzeysel",
        "coop": "İş birliği ruhu içinde",
        "positive": "Olumlu",
        "mutual_respect": "Karşılıklı saygıya dayalı",

        "cohab_stem": "Lütfen aşağıdaki ifadelere ne derece katıldığınızı veya katılmadığınızı belirtiniz.",
        "abs_disagree": "Kesinlikle katılmıyorum",
        "disagree": "Katılmıyorum",
        "neither": "Ne katılıyorum ne katılmıyorum",
        "agree": "Katılıyorum",
        "abs_agree": "Kesinlikle katılıyorum",
        "live_together": "{other} ile birlikte yaşayabileceğimi hissediyorum",
        "neighbors": "{other} komşum olursa rahatsız olmam",

        "thermo_stem": "Aşağıdaki soru genel olarak {other} hakkındaki duygularınızla ilgilidir. Lütfen bu grubu 0 ile 10 arasında değerlendiriniz. Daha yüksek puan daha sıcak veya olumlu hissettiğinizi gösterir. Daha düşük puan daha soğuk veya olumsuz hissettiğinizi gösterir. Ne sıcak ne soğuk hissediyorsanız 5'i seçiniz.",
        "thermo_q": "Genel olarak {other} hakkında nasıl hissediyorsunuz?",

        "trust_stem": "Şimdi genel olarak {other} hakkında bazı sorular sormak istiyoruz. Lütfen aşağıdaki ifadelere katılıp katılmadığınızı belirtiniz.",
        "trust_love": "{other} Kıbrıs'ı sevdiklerini söylediklerinde onlara güvenirim",
        "no_trust_politicians": "Anlaşılmış bir çözümün uygulanması söz konusu olduğunda {other_singular} siyasetçilere güvenmekte zorlanırım",
        "trust_ordinary": "Sıradan {other_singular} kişiler barış istediklerini söylediklerinde onlara güvenirim",
        "trust_politicians": "{other_singular} siyasetçiler barış istediklerini söylediklerinde onlara güvenirim",

        "threat_stem": "Aşağıdaki ifadelere ne derece katılıyor veya katılmıyorsunuz?",
        "rthreat_power": "{other} bu ülkede ne kadar fazla güç kazanırsa, {own} için işler o kadar zorlaşır",
        "rthreat_political": "{other} siyasi konularda karar verdiğinde, {own} bu ülkenin nasıl yönetildiği konusunda daha az söz sahibi olur",
        "rthreat_claim_more": "{other} gelecekte bizden giderek daha fazla şey talep edecek diye endişeleniyorum",
        "sthreat_values": "Kıbrıslı Türkler ve Kıbrıslı Rumlar Kıbrıs'ta çok farklı değerlere sahiptir",
        "sthreat_do_things": "{other} bazen {own} asla yapmayacağı şeyler yapar",

        "identity_stem": "Aşağıda bazı ifadeler yer almaktadır. Lütfen her ifadeye ne derece katıldığınızı veya katılmadığınızı belirtiniz.",
        "id_happy": "Genel olarak {own_singular} olmaktan mutluyum",
        "id_proud": "{own_singular} olmaktan gurur duyuyorum",
        "id_important": "{own_singular} olmam kimliğimin önemli bir parçasıdır",
        "id_self": "{own_singular} olmam kendimi nasıl gördüğümün önemli bir parçasıdır",

        "solution_stem": "Kıbrıs sorunu için aşağıdaki olası çözümlerin her biri için şu üç seçenekten hangisini seçerdiniz: karşı, destekliyorum veya ne karşı ne destekliyorum ama gerekirse tolere edebilirim?",
        "against": "Karşı",
        "tolerate": "Ne karşı ne destekliyorum, ama gerekirse tolere edebilirim",
        "in_favour": "Destekliyorum",
        "status_quo": "Mevcut durumun devamı",
        "bbf": "İki bölgeli, iki toplumlu federasyon",
        "unitary": "Üniter devlet",
        "two_states": "İki devletli çözüm"
    }
}

txt = T[lang]

# ============================================================
# HELPERS
# ============================================================

def go_next():
    st.session_state.page += 1
    st.rerun()

def go_back():
    st.session_state.page -= 1
    st.rerun()

def save_value(key, value):
    st.session_state.data[key] = value

def get_labels():
    community = st.session_state.data.get("community")

    if lang == "en":
        if community == 1:
            return {
                "own_plural": "Greek Cypriots",
                "own_singular": "Greek Cypriot",
                "other_plural": "Turkish Cypriots",
                "other_singular": "Turkish Cypriot",
                "motherland": "Greek",
                "displacement_place": "north of Cyprus",
                "displacement_events": "1974",
                "property_place": "north of Cyprus"
            }
        elif community == 2:
            return {
                "own_plural": "Turkish Cypriots",
                "own_singular": "Turkish Cypriot",
                "other_plural": "Greek Cypriots",
                "other_singular": "Greek Cypriot",
                "motherland": "Turkish",
                "displacement_place": "south of Cyprus",
                "displacement_events": "1963–1964 or 1974",
                "property_place": "south of Cyprus"
            }

    if lang == "el":
        if community == 1:
            return {
                "own_plural": "τους Ελληνοκύπριους",
                "own_plural_nom": "οι Ελληνοκύπριοι",
                "own_singular": "Ελληνοκύπριος/α",
                "other_plural": "τους Τουρκοκύπριους",
                "other_plural_nom": "οι Τουρκοκύπριοι",
                "other_singular": "Τουρκοκύπριους/ες",
                "motherland": "Έλληνας/Ελληνίδα",
                "displacement_place": "κατεχόμενη περιοχή της Κύπρου",
                "displacement_events": "1974",
                "property_place": "κατεχόμενη περιοχή της Κύπρου"
            }
        elif community == 2:
            return {
                "own_plural": "τους Τουρκοκύπριους",
                "own_plural_nom": "οι Τουρκοκύπριοι",
                "own_singular": "Τουρκοκύπριος/α",
                "other_plural": "τους Ελληνοκύπριους",
                "other_plural_nom": "οι Ελληνοκύπριοι",
                "other_singular": "Ελληνοκύπριους/ες",
                "motherland": "Τούρκος/Τουρκάλα",
                "displacement_place": "νότο της Κύπρου",
                "displacement_events": "1963–1964 ή 1974",
                "property_place": "νότο της Κύπρου"
            }

    if lang == "tr":
        if community == 1:
            return {
                "own_plural": "Kıbrıslı Rumlar",
                "own_singular": "Kıbrıslı Rum",
                "other_plural": "Kıbrıslı Türkler",
                "other_singular": "Kıbrıslı Türk",
                "motherland": "Yunan",
                "displacement_place": "Kıbrıs'ın kuzeyi",
                "displacement_events": "1974",
                "property_place": "Kıbrıs'ın kuzeyi"
            }
        elif community == 2:
            return {
                "own_plural": "Kıbrıslı Türkler",
                "own_singular": "Kıbrıslı Türk",
                "other_plural": "Kıbrıslı Rumlar",
                "other_singular": "Kıbrıslı Rum",
                "motherland": "Türk",
                "displacement_place": "Kıbrıs'ın güneyi",
                "displacement_events": "1963–1964 veya 1974",
                "property_place": "Kıbrıs'ın güneyi"
            }

    return {
        "own_plural": txt["community"],
        "own_plural_nom": txt["community"],
        "own_singular": txt["community"],
        "other_plural": txt["community"],
        "other_plural_nom": txt["community"],
        "other_singular": txt["community"],
        "motherland": "Greek/Turkish",
        "displacement_place": "",
        "displacement_events": "",
        "property_place": ""
    }

labels = get_labels()

def fmt(options):
    return lambda x: options[x]

tab_survey, tab_analysis = st.tabs(["Survey", "Representations Analysis"])

with tab_survey:
    show_logo_header()
    # ============================================================
    # HEADER
    # ============================================================

    st.title(txt["title"])
    st.caption(txt["caption"])
    st.markdown(f"### {txt['page']} {st.session_state.page} / {TOTAL_PAGES}")
    st.progress(st.session_state.page / TOTAL_PAGES)

    save_value("language", lang)

    # ============================================================
    # COMMON OPTION SETS
    # ============================================================

    yn_options = {
        1: txt["yes"],
        2: txt["no"],
        7: txt["refusal"],
        8: txt["dk"]
    }

    agree_scale = {
        1: txt["abs_disagree"],
        2: txt["disagree"],
        3: txt["neither"],
        4: txt["agree"],
        5: txt["abs_agree"],
        99: txt["dk_nr"]
    }

    # ============================================================
    # PAGE 1 — DEMOGRAPHICS
    # ============================================================

    if st.session_state.page == 1:

        st.header(txt["p1"])

        community_options = {
            1: txt["gc"],
            2: txt["tc"]
        }

        gender_options = {
            1: txt["male"],
            2: txt["female"]
        }

        education_options = {
            1: txt["reads_writes"],
            2: txt["elementary"],
            3: txt["gymnasium"],
            4: txt["lyceum"],
            5: txt["college"],
            6: txt["university"],
            7: txt["postgrad"],
            99: txt["na_dk"]
        }

        urban_options = {
            1: txt["urban"],
            2: txt["rural"],
            7: txt["refusal"],
            8: txt["dk"]
        }

        community = st.selectbox(txt["community"], list(community_options.keys()), format_func=fmt(community_options), key="community_widget")
        age = st.number_input(txt["age"], min_value=15, max_value=120, value=st.session_state.data.get("age", 30), key="age_widget")
        yearborn = st.number_input(txt["yearborn"], min_value=1900, max_value=2015, value=st.session_state.data.get("yearborn", 1990), key="yearborn_widget")
        gender = st.selectbox(txt["gender"], list(gender_options.keys()), format_func=fmt(gender_options), key="gender_widget")
        education = st.selectbox(txt["education"], list(education_options.keys()), format_func=fmt(education_options), key="education_widget")
        urban = st.selectbox(txt["area"], list(urban_options.keys()), format_func=fmt(urban_options), key="urban_widget")

        if st.button(txt["next"], key="next_1"):
            save_value("community", community)
            save_value("age", age)
            save_value("yearborn", yearborn)
            save_value("gender", gender)
            save_value("education", education)
            save_value("urban", urban)
            save_value("language", lang)
            go_next()

    # ============================================================
    # PAGE 2 — MORENO IDENTITY
    # ============================================================

    elif st.session_state.page == 2:

        labels = get_labels()
        st.header(txt["p2"])
        st.markdown(txt["moreno_stem"])

        moreno_options = {
            1: txt["only_cypriot"].format(motherland=labels["motherland"]),
            2: txt["cypriot_bit"].format(motherland=labels["motherland"]),
            3: txt["equal_cy_mother"].format(motherland=labels["motherland"]),
            4: txt["mother_bit_cy"].format(motherland=labels["motherland"]),
            5: txt["only_mother"].format(motherland=labels["motherland"]),
            7: txt["refusal"],
            8: txt["dk"]
        }

        moreno_identity = st.selectbox(txt["p2"], list(moreno_options.keys()), format_func=fmt(moreno_options), key="moreno_identity_widget")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_2"):
                go_back()
        with col2:
            if st.button(txt["next"], key="next_2"):
                save_value("Moreno_identity", moreno_identity)
                go_next()

    # ============================================================
    # PAGE 3 — RELIGIOSITY
    # ============================================================

    elif st.session_state.page == 3:

        st.header(txt["p3"])

        religion_important_options = {
            1: txt["not_at_all"],
            2: txt["little_importance"],
            3: txt["important"],
            4: txt["very_important"],
            5: txt["extremely_important"],
            99: txt["nr"]
        }

        religion_practice_options = {
            1: txt["never_practically"],
            2: txt["once_year"],
            3: txt["once_month"],
            4: txt["once_two_weeks"],
            5: txt["once_week"],
            99: txt["nr"]
        }

        religion_important = st.selectbox(txt["religion_important"], list(religion_important_options.keys()), format_func=fmt(religion_important_options), key="religion_important_widget")
        religion_practice = st.selectbox(txt["religion_practice"], list(religion_practice_options.keys()), format_func=fmt(religion_practice_options), key="religion_practice_widget")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_3"):
                go_back()
        with col2:
            if st.button(txt["next"], key="next_3"):
                save_value("Religion_Important", religion_important)
                save_value("Religion_Practice", religion_practice)
                go_next()

    # ============================================================
    # PAGE 4 — IDP / PROPERTY
    # ============================================================

    elif st.session_state.page == 4:

        labels = get_labels()
        st.header(txt["p4"])

        s4_idp_self = st.selectbox(
            txt["idp_self"].format(place=labels["displacement_place"], events=labels["displacement_events"]),
            list(yn_options.keys()),
            format_func=fmt(yn_options),
            key="s4_idp_self_widget"
        )

        s5_idp_family = st.selectbox(
            txt["idp_family"].format(place=labels["displacement_place"], events=labels["displacement_events"]),
            list(yn_options.keys()),
            format_func=fmt(yn_options),
            key="s5_idp_family_widget"
        )

        s6_property_north = st.selectbox(
            txt["property"].format(place=labels["property_place"]),
            list(yn_options.keys()),
            format_func=fmt(yn_options),
            key="s6_property_north_widget"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_4"):
                go_back()
        with col2:
            if st.button(txt["next"], key="next_4"):
                save_value("s4_idp_self", s4_idp_self)
                save_value("s5_idp_family", s5_idp_family)
                save_value("s6_property_north", s6_property_north)
                go_next()

    # ============================================================
    # PAGE 5 — CONTACT FREQUENCY
    # ============================================================

    elif st.session_state.page == 5:

        labels = get_labels()
        st.header(txt["p5"])
        st.markdown(txt["contact_stem"].format(other=labels["other_plural"]))

        contact_scale = {
            1: txt["never"],
            2: txt["less_month"],
            3: txt["once_month_contact"],
            4: txt["several_month"],
            5: txt["once_week_contact"],
            6: txt["several_week"],
            7: txt["every_day"],
            77: txt["refusal"],
            88: txt["dk"]
        }

        def contact_question(label, key):
            return st.selectbox(label, list(contact_scale.keys()), format_func=fmt(contact_scale), key=key)

        s7_contact_work = contact_question(txt["work"], "s7_contact_work_widget")
        s7_contact_bicommunal = contact_question(txt["bicommunal"], "s7_contact_bicommunal_widget")
        s7_contact_neighbourhood = contact_question(txt["neighbourhood"], "s7_contact_neighbourhood_widget")
        s7_contact_occupied_areas = contact_question(txt["occupied_gc"] if st.session_state.data.get("community") == 1 else txt["occupied_tc"], "s7_contact_occupied_areas_widget")
        s7_contact_non_occupied_areas = contact_question(txt["nonoccupied_gc"] if st.session_state.data.get("community") == 1 else txt["nonoccupied_tc"], "s7_contact_non_occupied_areas_widget")
        s7_contact_social_media = contact_question(txt["social_media"], "s7_contact_social_media_widget")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_5"):
                go_back()
        with col2:
            if st.button(txt["next"], key="next_5"):
                save_value("s7_contact_work", s7_contact_work)
                save_value("s7_contact_bicommunal", s7_contact_bicommunal)
                save_value("s7_contact_neighbourhood", s7_contact_neighbourhood)
                save_value("s7_contact_occupied_areas", s7_contact_occupied_areas)
                save_value("s7_contact_non_occupied_areas", s7_contact_non_occupied_areas)
                save_value("s7_contact_social_media", s7_contact_social_media)
                go_next()

    # ============================================================
    # PAGE 6 — CONTACT QUALITY
    # ============================================================

    elif st.session_state.page == 6:

        labels = get_labels()
        st.header(txt["p6"])
        st.markdown(txt["quality_stem"].format(other=labels["other_plural"]))

        quality_scale = {
            1: txt["not_at_all"],
            2: txt["little_bit"],
            3: txt["moderately"],
            4: txt["quite_bit"],
            5: txt["very_much"],
            99: txt["nr"]
        }

        def quality_question(label, key):
            return st.selectbox(label, list(quality_scale.keys()), format_func=fmt(quality_scale), key=key)

        qual_pleasant = quality_question(txt["pleasant"], "qual_pleasant_widget")
        qual_superficial = quality_question(txt["superficial"], "qual_superficial_widget")
        qual_coop = quality_question(txt["coop"], "qual_coop_widget")
        qual_positive = quality_question(txt["positive"], "qual_positive_widget")
        qual_mutual_respect = quality_question(txt["mutual_respect"], "qual_mutual_respect_widget")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_6"):
                go_back()
        with col2:
            if st.button(txt["next"], key="next_6"):
                save_value("Qual_Pleasant", qual_pleasant)
                save_value("Qual_Superficial", qual_superficial)
                save_value("Qual_Coop", qual_coop)
                save_value("Qual_Positive", qual_positive)
                save_value("Qual_Mutual_Respect", qual_mutual_respect)
                go_next()

    # ============================================================
    # PAGE 7 — COHABITATION
    # ============================================================

    elif st.session_state.page == 7:

        labels = get_labels()
        st.header(txt["p7"])
        st.markdown(txt["cohab_stem"])

        live_together = st.selectbox(
            txt["live_together"].format(other=labels["other_plural"]),
            list(agree_scale.keys()),
            format_func=fmt(agree_scale),
            key="live_together_widget"
        )

        not_mind_neighbors = st.selectbox(
            txt["neighbors"].format(other=labels["other_plural"]),
            list(agree_scale.keys()),
            format_func=fmt(agree_scale),
            key="not_mind_neighbors_widget"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_7"):
                go_back()
        with col2:
            if st.button(txt["next"], key="next_7"):
                save_value("Live_together", live_together)
                save_value("Not_mind_neighbors", not_mind_neighbors)
                go_next()

    # ============================================================
    # PAGE 8 — THERMOMETER
    # ============================================================

    elif st.session_state.page == 8:

        labels = get_labels()
        st.header(txt["p8"])
        st.markdown(txt["thermo_stem"].format(other=labels["other_plural"]))

        s8_thermo_og_0_10 = st.slider(
            txt["thermo_q"].format(other=labels["other_plural"]),
            min_value=0,
            max_value=10,
            value=st.session_state.data.get("s8_thermo_og_0_10", 5),
            key="s8_thermo_widget"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_8"):
                go_back()
        with col2:
            if st.button(txt["next"], key="next_8"):
                save_value("s8_thermo_og_0_10", s8_thermo_og_0_10)
                go_next()

    # ============================================================
    # PAGE 9 — TRUST
    # ============================================================

    elif st.session_state.page == 9:

        labels = get_labels()
        st.header(txt["p9"])
        st.markdown(txt["trust_stem"].format(other=labels["other_plural"]))

        trust_love = st.selectbox(txt["trust_love"].format(other=labels["other_plural"]), list(agree_scale.keys()), format_func=fmt(agree_scale), key="trust_love_widget")
        no_trust_politicians = st.selectbox(txt["no_trust_politicians"].format(other_singular=labels["other_singular"]), list(agree_scale.keys()), format_func=fmt(agree_scale), key="no_trust_politicians_widget")
        trust_ordinary = st.selectbox(txt["trust_ordinary"].format(other_singular=labels["other_singular"]), list(agree_scale.keys()), format_func=fmt(agree_scale), key="trust_ordinary_widget")
        trust_politicians = st.selectbox(txt["trust_politicians"].format(other_singular=labels["other_singular"]), list(agree_scale.keys()), format_func=fmt(agree_scale), key="trust_politicians_widget")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_9"):
                go_back()
        with col2:
            if st.button(txt["next"], key="next_9"):
                save_value("Trust_love", trust_love)
                save_value("No_trust_politicians", no_trust_politicians)
                save_value("Trust_ordinary", trust_ordinary)
                save_value("Trust_politicians", trust_politicians)
                go_next()

    # ============================================================
    # PAGE 10 — THREATS
    # ============================================================

    elif st.session_state.page == 10:

        labels = get_labels()
        st.header(txt["p10"])
        st.markdown(txt["threat_stem"])

        if lang == "el":
            other_power = labels["other_plural_nom"]
            own_power = labels["own_plural"]
            other_political = labels["other_plural"]
            own_political = labels["own_plural_nom"]
            other_claim = labels["other_plural_nom"]
            other_do = labels["other_plural_nom"]
            own_do = labels["own_plural_nom"]
        else:
            other_power = labels["other_plural"]
            own_power = labels["own_plural"]
            other_political = labels["other_plural"]
            own_political = labels["own_plural"]
            other_claim = labels["other_plural"]
            other_do = labels["other_plural"]
            own_do = labels["own_plural"]

        rthreat_power = st.selectbox(
            txt["rthreat_power"].format(other=other_power, own=own_power),
            list(agree_scale.keys()),
            format_func=fmt(agree_scale),
            key="rthreat_power_widget"
        )

        rthreat_political = st.selectbox(
            txt["rthreat_political"].format(other=other_political, own=own_political),
            list(agree_scale.keys()),
            format_func=fmt(agree_scale),
            key="rthreat_political_widget"
        )

        rthreat_claim_more = st.selectbox(
            txt["rthreat_claim_more"].format(other=other_claim),
            list(agree_scale.keys()),
            format_func=fmt(agree_scale),
            key="rthreat_claim_more_widget"
        )

        sthreat_values = st.selectbox(
            txt["sthreat_values"],
            list(agree_scale.keys()),
            format_func=fmt(agree_scale),
            key="sthreat_values_widget"
        )

        sthreat_do_things = st.selectbox(
            txt["sthreat_do_things"].format(other=other_do, own=own_do),
            list(agree_scale.keys()),
            format_func=fmt(agree_scale),
            key="sthreat_do_things_widget"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_10"):
                go_back()
        with col2:
            if st.button(txt["next"], key="next_10"):
                save_value("Rthreat_power", rthreat_power)
                save_value("Rthreat_political", rthreat_political)
                save_value("Rthreat_claim_more", rthreat_claim_more)
                save_value("Sthreat_values", sthreat_values)
                save_value("Sthreat_do_things", sthreat_do_things)
                go_next()

    # ============================================================
    # PAGE 11 — COMMUNITY IDENTITY
    # ============================================================

    elif st.session_state.page == 11:

        labels = get_labels()
        st.header(txt["p11"])
        st.markdown(txt["identity_stem"])

        id_happy = st.selectbox(txt["id_happy"].format(own_singular=labels["own_singular"]), list(agree_scale.keys()), format_func=fmt(agree_scale), key="id_happy_widget")
        id_proud = st.selectbox(txt["id_proud"].format(own_singular=labels["own_singular"]), list(agree_scale.keys()), format_func=fmt(agree_scale), key="id_proud_widget")
        id_important = st.selectbox(txt["id_important"].format(own_singular=labels["own_singular"]), list(agree_scale.keys()), format_func=fmt(agree_scale), key="id_important_widget")
        id_self = st.selectbox(txt["id_self"].format(own_singular=labels["own_singular"]), list(agree_scale.keys()), format_func=fmt(agree_scale), key="id_self_widget")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_11"):
                go_back()
        with col2:
            if st.button(txt["next"], key="next_11"):
                save_value("Id_Happy", id_happy)
                save_value("Id_Proud", id_proud)
                save_value("Id_Important", id_important)
                save_value("Id_Self", id_self)
                go_next()

    # ============================================================
    # PAGE 12 — SOLUTIONS
    # ============================================================

    elif st.session_state.page == 12:

        st.header(txt["p12"])
        st.markdown(txt["solution_stem"])

        s3_options = {
            1: txt["against"],
            2: txt["tolerate"],
            3: txt["in_favour"],
            7: txt["refusal"],
            8: txt["dk"]
        }

        s3_status_quo = st.selectbox(txt["status_quo"], list(s3_options.keys()), format_func=fmt(s3_options), key="s3_status_quo_widget")
        s3_bbf = st.selectbox(txt["bbf"], list(s3_options.keys()), format_func=fmt(s3_options), key="s3_bbf_widget")
        s3_unitary = st.selectbox(txt["unitary"], list(s3_options.keys()), format_func=fmt(s3_options), key="s3_unitary_widget")
        s3_two_states = st.selectbox(txt["two_states"], list(s3_options.keys()), format_func=fmt(s3_options), key="s3_two_states_widget")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["back"], key="back_12"):
                go_back()
        with col2:
            if st.button(txt["submit"], key="submit_12"):
                save_value("s3_status_quo", s3_status_quo)
                save_value("s3_bbf", s3_bbf)
                save_value("s3_unitary", s3_unitary)
                save_value("s3_two_states", s3_two_states)
                save_value("consent_given", True)
                save_value("language", lang)

                end_time = datetime.now(timezone.utc)
                duration_seconds = (end_time - st.session_state.start_time).total_seconds()

                save_value("start_time", st.session_state.start_time.isoformat())
                save_value("end_time", end_time.isoformat())
                save_value("duration_seconds", duration_seconds)

                # Keep a copy of the most recent completed survey response.
                # The Analysis tab uses this to project the respondent into the historical representational space.
                st.session_state["latest_respondent"] = st.session_state.data.copy()

                try:
                    response = supabase.table("responses_raw").insert(st.session_state.data).execute()
                except Exception:
                    st.error(
                        "The response could not be saved because the app could not connect to Supabase. "
                        "Check Streamlit Cloud secrets: SUPABASE_URL should be the project API URL "
                        "(https://your-project-ref.supabase.co), and SUPABASE_KEY should be the anon key."
                    )
                    st.stop()

                st.success(txt["success"])
                st.write(txt["saved_response"])
                st.json(response.data)

                st.session_state.data = {}
                st.session_state.page = 1
                st.session_state.start_time = datetime.now(timezone.utc)



# ============================================================
# CURRENT RESPONDENT PROJECTION HELPERS
# ============================================================

def _clean_current_value(value):
    """Convert special survey missing codes to NaN while leaving valid numeric values untouched."""
    if value in [77, 88, 99, None]:
        return np.nan
    return value


def _mean_from_response(response, keys):
    vals = [_clean_current_value(response.get(k)) for k in keys]
    vals = [v for v in vals if not pd.isna(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def build_current_scales(response, historical_df=None):
    """Build scale-level scores for the latest survey response using the same constructs as the historical model."""
    current = {}
    current["community"] = response.get("community")

    current["Contact_Frequency"] = _mean_from_response(response, [
        "s7_contact_work", "s7_contact_bicommunal", "s7_contact_neighbourhood",
        "s7_contact_occupied_areas", "s7_contact_non_occupied_areas"
    ])
    current["Contact_Quality"] = _mean_from_response(response, [
        "Qual_Pleasant", "Qual_Coop", "Qual_Positive", "Qual_Mutual_Respect"
    ])
    current["Cohabitation"] = _mean_from_response(response, [
        "Live_together", "Not_mind_neighbors"
    ])
    current["Trust"] = _mean_from_response(response, [
        "Trust_love", "Trust_ordinary", "Trust_politicians"
    ])
    current["Threats"] = _mean_from_response(response, [
        "Rthreat_power", "Rthreat_political", "Rthreat_claim_more",
        "Sthreat_values", "Sthreat_do_things"
    ])
    current["Identity"] = _mean_from_response(response, [
        "Id_Happy", "Id_Proud", "Id_Important", "Id_Self"
    ])
    current["Religiosity"] = _mean_from_response(response, [
        "Religion_Important", "Religion_Practice"
    ])

    thermo = _clean_current_value(response.get("s8_thermo_og_0_10"))
    if not pd.isna(thermo):
        # The app asks the thermometer on a 0–10 scale. Historical files may code it as
        # 0–100, 1–11, or 0–10. Harmonise the current respondent to the historical scale.
        if historical_df is not None and "Thermometer" in historical_df.columns:
            hist_min = historical_df["Thermometer"].min(skipna=True)
            hist_max = historical_df["Thermometer"].max(skipna=True)
            if pd.notna(hist_min) and pd.notna(hist_max):
                if hist_max > 20:
                    thermo = thermo * 10
                elif hist_min >= 1 and hist_max > 10:
                    thermo = thermo + 1
        current["Thermometer"] = float(thermo)

    # Convert current survey solution preferences to binary acceptability:
    # 1 = accept/tolerate, 0 = reject. Survey coding: 1 = against, 2 = tolerate, 3 = in favour.
    solution_response_map = {
        "Solution_StatusQuo": "s3_status_quo",
        "Solution_BBF": "s3_bbf",
        "Solution_Unitary": "s3_unitary",
        "Solution_TwoStates": "s3_two_states",
    }
    for out_name, in_name in solution_response_map.items():
        val = _clean_current_value(response.get(in_name))
        if pd.isna(val):
            current[out_name] = np.nan
        elif val == 1:
            current[out_name] = 0.0
        elif val in [2, 3]:
            current[out_name] = 1.0
        else:
            current[out_name] = np.nan

    if "Moreno_identity" in response:
        current["Moreno"] = _clean_current_value(response.get("Moreno_identity"))

    return current



# ============================================================
# CLUSTER INTERPRETATION HELPERS
# ============================================================

def _safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def solution_orientation(row):
    """Interpret binary solution acceptability profile. Values are proportions in the cluster."""
    bbf = _safe_float(row.get("Solution_BBF", np.nan))
    two = _safe_float(row.get("Solution_TwoStates", np.nan))
    unitary = _safe_float(row.get("Solution_Unitary", np.nan))
    status = _safe_float(row.get("Solution_StatusQuo", np.nan))

    HIGH = 0.70
    MID = 0.30

    # Accepting two states is not necessarily partitionism if BBF is also accepted.
    if bbf >= HIGH and two >= HIGH:
        return "Multi-solution permissive / pragmatic"
    if bbf >= HIGH and two < MID:
        return "Federalist / BBF-oriented"
    if unitary >= HIGH and two < MID and bbf >= MID:
        return "Unitary / anti-partition with some federal tolerance"
    if unitary >= HIGH and two < MID and bbf < MID:
        return "Unitary / anti-partition"
    if two >= HIGH and bbf < MID:
        return "Partition-leaning"
    if status >= HIGH and bbf < MID and two < MID:
        return "Status-quo oriented"
    if bbf >= MID and two < MID:
        return "BBF-tolerant / anti-partition"
    return "Mixed / non-exclusive"


def interpret_cluster_profiles(profile, community_label, demographic_context=None):
    """Create community-sensitive GSP labels and paragraph narratives from cluster means."""
    out = profile.copy()
    positive_candidates = ["Trust", "Contact_Quality", "Contact_Frequency", "Cohabitation", "Thermometer"]
    negative_candidates = ["Threats"]

    index_parts = []
    for col in positive_candidates:
        if col in out.columns and out[col].nunique(dropna=True) > 1:
            sd = out[col].std(ddof=0)
            if sd and not pd.isna(sd):
                index_parts.append((out[col] - out[col].mean()) / sd)
    for col in negative_candidates:
        if col in out.columns and out[col].nunique(dropna=True) > 1:
            sd = out[col].std(ddof=0)
            if sd and not pd.isna(sd):
                index_parts.append(-1 * (out[col] - out[col].mean()) / sd)

    if index_parts:
        out["intergroup_index"] = pd.concat(index_parts, axis=1).mean(axis=1)
    else:
        out["intergroup_index"] = 0.0

    def intergroup_label(x):
        if x >= 0.45:
            return "Pro-reconciliation"
        if x <= -0.45:
            return "Ethno-nationalist / high-threat"
        return "Communitarian / ambivalent"

    out["intergroup_orientation"] = out["intergroup_index"].apply(intergroup_label)
    out["solution_orientation"] = out.apply(solution_orientation, axis=1)
    out["final_label"] = out["intergroup_orientation"] + " – " + out["solution_orientation"]

    is_tc = "Turkish" in community_label
    narratives = []
    demographic_context = demographic_context or {}

    for cluster, row in out.iterrows():
        inter = row["intergroup_orientation"]
        sol = row["solution_orientation"]

        if inter == "Pro-reconciliation":
            base = "This cluster is closest to the pro-reconciliation position: comparatively warmer feelings, more trust/contact or lower threat, and greater openness to coordination with the other community."
        elif inter == "Ethno-nationalist / high-threat":
            base = "This cluster is closest to the ethno-nationalist or high-threat position: comparatively colder feelings, weaker trust/contact or stronger threat perceptions, and a more defensive orientation toward the other community."
        else:
            if is_tc:
                base = "This cluster resembles a communitarian/autonomy-oriented position: not simply hostile, but marked by symbolic distance, concern with communal security or political autonomy, and a more conditional orientation toward cooperation."
            else:
                base = "This cluster resembles a communitarian/ambivalent position: not fully reconciliation-oriented, but not necessarily ethnonationalist; it often combines moderate intergroup evaluations with concerns about security, political equality, or the terms of coexistence."

        if sol == "Multi-solution permissive / pragmatic":
            sol_text = " Its solution profile is non-exclusive: it accepts multiple, even mutually incompatible, options. This should be read as pragmatic flexibility, tolerance, uncertainty, or low ideological exclusivity rather than as straightforward partitionism."
        elif sol == "Federalist / BBF-oriented":
            sol_text = " Its solution profile gives primary legitimacy to BBF/federal settlement and rejects separation-oriented options."
        elif "Unitary" in sol:
            sol_text = " Its solution profile privileges a unitary-state frame and resists partition-oriented outcomes."
        elif sol == "Partition-leaning":
            sol_text = " Its solution profile gives comparatively greater legitimacy to separation/two-state outcomes, especially where BBF acceptance is weak."
        elif sol == "Status-quo oriented":
            sol_text = " Its solution profile is relatively tolerant of maintaining the present situation."
        elif sol == "BBF-tolerant / anti-partition":
            sol_text = " Its solution profile keeps BBF within the range of acceptable compromise while resisting two-state separation."
        else:
            sol_text = " Its solution profile is mixed and should be interpreted cautiously rather than forced into a single ideological type."

        demographic_notes = demographic_context.get(str(cluster), [])
        if demographic_notes:
            demo_text = (
                " In the chi-square tests, this cluster is also "
                + "; ".join(demographic_notes[:4])
                + "."
            )
        else:
            demo_text = ""

        narratives.append(base + sol_text + demo_text)

    out["narrative"] = narratives
    return out


def show_current_respondent_position(current_cluster, current_probability, current_scores, interpreted_profile):
    """Display a respondent-facing explanation of the cluster assignment."""
    if current_cluster not in interpreted_profile.index:
        return
    row = interpreted_profile.loc[current_cluster]
    st.markdown("### Your position in the representational space")
    st.info(
        f"Your responses are closest to **Cluster {current_cluster}** "
        f"(**assignment probability: {current_probability:.2f}**).\n\n"
        f"**Suggested interpretation:** {row['final_label']}\n\n"
        f"{row['narrative']}"
    )

    solution_cols = ["Solution_StatusQuo", "Solution_BBF", "Solution_Unitary", "Solution_TwoStates"]
    available_solution_cols = [c for c in solution_cols if c in current_scores]
    if available_solution_cols:
        readable = {
            "Solution_StatusQuo": "Status quo",
            "Solution_BBF": "BBF",
            "Solution_Unitary": "Unitary state",
            "Solution_TwoStates": "Two states",
        }
        solution_df = pd.DataFrame([
            {
                "solution": readable[c],
                "your_response": "Accept / tolerate" if current_scores.get(c) == 1 else "Reject" if current_scores.get(c) == 0 else "Missing",
                "cluster_acceptance_percent": round(100 * _safe_float(row.get(c, np.nan)), 1) if c in row else np.nan,
            }
            for c in available_solution_cols
        ])
        with st.expander("Your solution preferences compared with your closest cluster", expanded=True):
            st.dataframe(solution_df, use_container_width=True)


def load_permanent_historical_data():
    for path in HISTORICAL_DATA_CANDIDATES:
        if not path.exists():
            continue

        if path.suffix.lower() == ".sav":
            df, meta = pyreadstat.read_sav(path)
            return df, path, getattr(meta, "variable_value_labels", {}) or {}

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path), path, {}

    return None, None, {}


def fetch_supabase_responses():
    rows = []
    page_size = 1000
    start = 0

    try:
        while True:
            response = (
                supabase.table("responses_raw")
                .select("*")
                .range(start, start + page_size - 1)
                .execute()
            )
            page = response.data or []
            rows.extend(page)
            if len(page) < page_size:
                break
            start += page_size
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def infer_response_period(response):
    for key in ["end_time", "start_time", "created_at", "inserted_at"]:
        value = response.get(key)
        if value is None or pd.isna(value):
            continue

        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.notna(timestamp):
            return int(timestamp.year)

    return datetime.now(timezone.utc).year


def build_new_response_analysis_rows(raw_responses, historical_df):
    if raw_responses.empty:
        return pd.DataFrame()

    rows = []
    for response in raw_responses.to_dict(orient="records"):
        current = build_current_scales(response, historical_df=historical_df)
        if current.get("community") not in [1, 2]:
            continue

        current["period"] = infer_response_period(response)
        current["source"] = "New questionnaire"
        rows.append(current)

    return pd.DataFrame(rows)


def find_first_existing_column(dataframe, candidates):
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def clean_categorical_for_chi_square(series):
    cleaned = series.copy()
    cleaned = cleaned.replace({77: np.nan, 88: np.nan, 99: np.nan})
    cleaned = cleaned.dropna()
    cleaned = cleaned[~cleaned.astype(str).str.strip().str.lower().isin(["", "nan", "none", "dk", "nr", "dk/nr", "dk / nr"])]
    return cleaned


def compact_category_value(value):
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric_value) and float(numeric_value).is_integer():
        return str(int(numeric_value))
    return str(value).strip()


def value_label_lookup(value_labels, column):
    if not value_labels:
        return {}

    candidates = [column]
    if column == "period":
        candidates.append("Period")
    elif column == "community":
        candidates.append("GCs")

    for candidate in candidates:
        if candidate in value_labels:
            return value_labels[candidate] or {}

    return {}


def labelled_category_series(series, label_map):
    cleaned = clean_categorical_for_chi_square(series)

    def label_one(value):
        raw = compact_category_value(value)
        labels_to_try = [
            value,
            pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0],
            raw,
        ]

        for candidate in labels_to_try:
            if candidate in label_map:
                return f"{raw} = {label_map[candidate]}"

        return raw

    return cleaned.apply(label_one)


def recode_generation_for_chi_square(dataframe):
    yearborn_col = find_first_existing_column(dataframe, ["yearborn", "Yearborn", "YearBorn", "Year born"])
    age_col = find_first_existing_column(dataframe, ["Age", "age"])
    period_col = find_first_existing_column(dataframe, ["period", "Period"])

    if yearborn_col:
        birth_year = pd.to_numeric(dataframe[yearborn_col], errors="coerce")
    elif age_col and period_col:
        birth_year = (
            pd.to_numeric(dataframe[period_col], errors="coerce")
            - pd.to_numeric(dataframe[age_col], errors="coerce")
        )
    else:
        return pd.Series(index=dataframe.index, dtype="object")

    generation = pd.cut(
        birth_year,
        bins=[0, 1945, 1964, 1980, 1996, 2100],
        labels=[
            "Born up to 1945",
            "Born 1946-1964",
            "Born 1965-1980",
            "Born 1981-1996",
            "Born 1997 or later",
        ],
        right=True,
    )
    return generation.astype("object")


def chi_square_summary(table):
    chi2, p_value, dof, expected = chi2_contingency(table)
    total = table.to_numpy().sum()
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = (chi2 / (total * min_dim)) ** 0.5 if total and min_dim else 0.0
    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "cramers_v": cramers_v,
        "min_expected": float(np.min(expected)),
        "expected": expected,
    }


def render_cluster_demographic_chi_square(source_df, clustered_df, community_label, value_labels=None):
    st.write("Cluster differences by demographic variables")
    st.caption(
        "Chi-square tests examine whether the distribution of model-derived clusters differs "
        "across demographic categories. Generation is recoded from year of birth when available, "
        "or estimated from period minus age. Raw year of birth is not tested because it has too many categories."
    )

    demographic_sources = {
        "Origin": find_first_existing_column(source_df, ["Origin"]),
        "Period": find_first_existing_column(source_df, ["period", "Period"]),
        "Generation": "Generation",
        "Male": find_first_existing_column(source_df, ["Male"]),
        "Education": find_first_existing_column(source_df, ["Education"]),
        "Urban": find_first_existing_column(source_df, ["Urban"]),
        "IDP1_2": find_first_existing_column(source_df, ["IDP1_2"]),
    }

    demographic_frame = source_df.loc[clustered_df.index].copy()
    demographic_frame["Generation"] = recode_generation_for_chi_square(demographic_frame)
    demographic_frame["cluster"] = clustered_df["cluster"].astype(str)

    available_demographics = {
        label: column
        for label, column in demographic_sources.items()
        if column is not None and column in demographic_frame.columns
    }

    if not available_demographics:
        st.info("No expected demographic variables were found for chi-square testing.")
        return {}

    selected_demographics = st.multiselect(
        f"Demographic variables for chi-square tests - {community_label}",
        list(available_demographics.keys()),
        default=list(available_demographics.keys()),
        key=f"chi_square_demographics_{community_label}",
    )

    if not selected_demographics:
        st.info("Select at least one demographic variable to run chi-square tests.")
        return {}

    summary_rows = []
    tables = {}
    row_percentages = {}
    demographic_context = {}

    for label in selected_demographics:
        column = available_demographics[label]
        label_map = value_label_lookup(value_labels, column)
        analysis_data = pd.DataFrame(
            {
                "cluster": demographic_frame["cluster"],
                "demographic": labelled_category_series(demographic_frame[column], label_map),
            }
        ).dropna()

        if analysis_data["cluster"].nunique() < 2 or analysis_data["demographic"].nunique() < 2:
            continue

        table = pd.crosstab(analysis_data["demographic"], analysis_data["cluster"])
        result = chi_square_summary(table)

        summary_rows.append(
            {
                "Variable": label,
                "N": int(table.to_numpy().sum()),
                "Chi-square": round(result["chi2"], 3),
                "df": int(result["dof"]),
                "p-value": round(result["p_value"], 4),
                "Cramer's V": round(result["cramers_v"], 3),
                "Minimum expected count": round(result["min_expected"], 2),
            }
        )
        tables[label] = table
        row_percentages[label] = (100 * table.div(table.sum(axis=1), axis=0)).round(1)

        if result["p_value"] < 0.05:
            observed = table.to_numpy()
            expected = result["expected"]
            residuals = (observed - expected) / np.sqrt(expected)

            for row_index, category in enumerate(table.index):
                for col_index, cluster in enumerate(table.columns):
                    residual = residuals[row_index, col_index]
                    if residual >= 1.96:
                        demographic_context.setdefault(str(cluster), []).append(
                            f"over-represented among {label}: {category}"
                        )
                    elif residual <= -1.96:
                        demographic_context.setdefault(str(cluster), []).append(
                            f"under-represented among {label}: {category}"
                        )

    if not summary_rows:
        st.info("The selected variables did not have enough valid categories for chi-square tests.")
        return {}

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    if (summary_df["Minimum expected count"] < 5).any():
        st.warning(
            "At least one test has expected cell counts below 5. Interpret those chi-square results cautiously."
        )

    for label in tables:
        st.markdown(f"**{label}: observed counts**")
        st.dataframe(tables[label], use_container_width=True)

        st.markdown(f"**{label}: row percentages**")
        st.dataframe(row_percentages[label], use_container_width=True)

        plot_data = row_percentages[label].reset_index().melt(
            id_vars=row_percentages[label].index.name or "demographic",
            var_name="Cluster",
            value_name="Percent",
        )
        plot_data = plot_data.rename(columns={plot_data.columns[0]: label})

        fig_chi = px.bar(
            plot_data,
            x=label,
            y="Percent",
            color="Cluster",
            barmode="group",
            title=f"{label} by cluster - {community_label}",
        )
        st.plotly_chart(fig_chi, use_container_width=True)

    return demographic_context


with tab_analysis:

    show_logo_header()
    st.header("Cyprus Issue Representations Analysis")

    st.markdown(
        """
        The representations analysis uses the permanent historical dataset bundled with this app
        and enriches it with new questionnaire responses saved in Supabase.

        The bundled file should be named `data/historical_responses.sav` or
        `data/historical_responses.csv`.
        """
    )

    historical_df, historical_source, historical_value_labels = load_permanent_historical_data()

    if historical_df is None:
        st.error(
            "No permanent historical dataset was found. Add `data/historical_responses.sav` "
            "or `data/historical_responses.csv` to the GitHub repository."
        )
        st.stop()

    if historical_df is not None:
        df = historical_df.copy()

        st.success(f"Historical dataset loaded from `{historical_source}`.")

        with st.expander("Preview uploaded data", expanded=False):
            st.dataframe(df.head())

        with st.expander("Detected variables", expanded=False):
            st.write(list(df.columns))

        # ----------------------------------------------------
        # Harmonise base variables
        # ----------------------------------------------------
        if "GCs" in df.columns:
            df = df.rename(columns={"GCs": "community"})
        else:
            st.error("Variable `GCs` not found. Please check the SPSS variable name for community.")
            st.stop()

        if "Period" in df.columns:
            df = df.rename(columns={"Period": "period"})
        else:
            st.error("Variable `Period` not found. Please check the SPSS variable name for survey wave/year.")
            st.stop()

        # Recode community to match the survey app convention:
        # App: 1 = Greek Cypriot, 2 = Turkish Cypriot
        # SPSS: 1 = Greek Cypriot, 0 = Turkish Cypriot
        df["community"] = df["community"].replace({1: 1, 0: 2})

        # ----------------------------------------------------
        # Clean common special missing values
        # ----------------------------------------------------
        df = df.replace({77: np.nan, 88: np.nan, 99: np.nan})

        # ----------------------------------------------------
        # Scale construction
        # ----------------------------------------------------
        st.subheader("Scale construction")

        scale_map = {
            "Contact_Frequency": [
                "Contact_work",
                "Contact_Bicommunal",
                "Contact_area",
                "Contact_other_side",
                "Contact_this_side"
            ],
            "Contact_Quality": [
                "Qual_Pleasant",
                "Qual_Coop",
                "Qual_Positive",
                "Qual_Mutual_Respect"
            ],
            "Cohabitation": [
                "Live_together",
                "Not_mind_neighbors"
            ],
            "Trust": [
                "Trust_love",
                "Trust_ordinary",
                "Trust_politicians"
            ],
            "Threats": [
                "Rthreat_power",
                "Rthreat_political",
                "Rthreat_claim_more",
                "Sthreat_values",
                "Sthreat_do_things"
            ],
            "Identity": [
                "Comm_ID_Happy",
                "Comm_ID_Proud",
                "Comm_ID_Imp1",
                "Comm_ID_Imp2"
            ],
            "Religiosity": [
                "Religion_Important",
                "Religion_Practice"
            ]
        }

        constructed_scales = []

        for scale_name, items in scale_map.items():
            available = [v for v in items if v in df.columns]

            if len(available) >= 2:
                df[scale_name] = df[available].mean(axis=1)
                constructed_scales.append(scale_name)
                st.write(f"✅ {scale_name}: {available}")
            else:
                st.warning(f"⚠️ {scale_name} not constructed. Found variables: {available}")

        # Single-item variables
        if "ThermoOG" in df.columns:
            df["Thermometer"] = df["ThermoOG"]
            constructed_scales.append("Thermometer")
            st.write("✅ Thermometer: ThermoOG")
        else:
            st.info("Thermometer not constructed because `ThermoOG` was not found.")

        # Moreno is a new item in the survey tab and is not available in the 2007–2017 SPSS file.
        if "Moreno_identity" in df.columns:
            df["Moreno"] = df["Moreno_identity"]
            constructed_scales.append("Moreno")
            st.write("✅ Moreno: Moreno_identity")
        else:
            st.info("Moreno is not available in the historical SPSS file. It will be available for new survey data only.")

        # ----------------------------------------------------
        # Solution-preference variables
        # ----------------------------------------------------
        st.subheader("Solution-preference variables")
        st.caption("Binary convention: 1 = accept/tolerate, 0 = reject. Cluster means are proportions accepting/tolerating each solution.")

        solution_var_map = {
            "Solution_StatusQuo": "AccStatus_Quo",
            "Solution_BBF": "AccBBF",
            "Solution_Unitary": "AccUnitary",
            "Solution_TwoStates": "AccTwo_States",
        }

        for scale_name, source_col in solution_var_map.items():
            if source_col in df.columns:
                df[scale_name] = pd.to_numeric(df[source_col], errors="coerce")
                constructed_scales.append(scale_name)
                st.write(f"✅ {scale_name}: {source_col}")
            else:
                st.info(f"{scale_name} not constructed. Variable `{source_col}` was not found.")

        if len(constructed_scales) < 3:
            st.error("Too few scales were constructed. We need to map the exact SPSS variable names.")
            st.stop()

        df["source"] = "Historical file"
        new_response_rows = build_new_response_analysis_rows(fetch_supabase_responses(), historical_df=df)
        if not new_response_rows.empty:
            df = pd.concat([df, new_response_rows], ignore_index=True, sort=False)
            st.success(f"Added {len(new_response_rows)} questionnaire response(s) from Supabase.")
        else:
            st.info("No saved questionnaire responses were available from Supabase yet.")

        st.success("Scale construction completed.")

        # ----------------------------------------------------
        # Clustering settings
        # ----------------------------------------------------
        st.subheader("Clustering settings")

        features = st.multiselect(
            "Select variables for clustering",
            constructed_scales,
            default=constructed_scales
        )

        latest_response = st.session_state.get("latest_respondent")
        if latest_response is None:
            st.info("No completed survey response is available yet. Complete Tab 1 and submit it to show the 'You are here' marker.")
        else:
            st.success("A completed survey response is available for projection. It will be shown as 'Current respondent' when the required variables match the selected clustering variables.")

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            min_k = st.number_input("Minimum number of clusters", min_value=2, max_value=6, value=2, key="min_k")
        with col_k2:
            max_k = st.number_input("Maximum number of clusters", min_value=3, max_value=10, value=6, key="max_k")

        if max_k <= min_k:
            st.error("Maximum number of clusters must be larger than minimum number of clusters.")
            st.stop()

        run_analysis = st.button("Run representations analysis", key="run_representations_analysis")

        if run_analysis:
            if not features:
                st.error("Please select at least one variable for clustering.")
                st.stop()

            analysis_df = df[["community", "period"] + features].dropna().copy()

            st.write("Cases available for analysis:", len(analysis_df))

            if len(analysis_df) < 100:
                st.warning("The available sample is small. Results may be unstable.")

            community_names = {
                1: "Greek Cypriots",
                2: "Turkish Cypriots"
            }

            for comm in sorted(analysis_df["community"].dropna().unique()):
                community_df = analysis_df[analysis_df["community"] == comm].copy()
                community_label = community_names.get(comm, f"Community {comm}")

                if len(community_df) < 50:
                    st.warning(f"{community_label}: too few cases for clustering.")
                    continue

                st.divider()
                st.subheader(community_label)

                X = community_df[features].copy()

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # ----------------------------------------------------
                # Automatic cluster selection using BIC
                # ----------------------------------------------------
                bic_scores = []
                models = {}

                for k in range(int(min_k), int(max_k) + 1):
                    gmm = GaussianMixture(
                        n_components=k,
                        covariance_type="full",
                        random_state=42
                    )
                    gmm.fit(X_scaled)
                    bic_scores.append({"k": k, "BIC": gmm.bic(X_scaled)})
                    models[k] = gmm

                bic_df = pd.DataFrame(bic_scores)
                best_k = int(bic_df.loc[bic_df["BIC"].idxmin(), "k"])
                best_model = models[best_k]

                st.write(f"Selected number of clusters by BIC: **{best_k}**")
                st.dataframe(bic_df, use_container_width=True)

                cluster_labels = best_model.predict(X_scaled)
                cluster_probs = best_model.predict_proba(X_scaled).max(axis=1)

                community_df["cluster"] = cluster_labels
                community_df["cluster_probability"] = cluster_probs

                # ----------------------------------------------------
                # Cluster sizes
                # ----------------------------------------------------
                st.write("Cluster sizes")

                cluster_sizes = community_df.groupby("cluster").size().reset_index(name="n")
                cluster_sizes["percent"] = (100 * cluster_sizes["n"] / cluster_sizes["n"].sum()).round(1)
                st.dataframe(cluster_sizes, use_container_width=True)

                # ----------------------------------------------------
                # Chi-square tests: clusters by demographic variables
                # ----------------------------------------------------
                demographic_context = render_cluster_demographic_chi_square(
                    df,
                    community_df,
                    community_label,
                    historical_value_labels,
                )

                # ----------------------------------------------------
                # Cluster profiles
                # ----------------------------------------------------
                st.write("Cluster profiles: mean scale scores")

                profile = community_df.groupby("cluster")[features].mean().round(3)
                st.dataframe(profile, use_container_width=True)

                interpreted_profile = interpret_cluster_profiles(
                    profile,
                    community_label,
                    demographic_context=demographic_context,
                )
                st.write("Cluster interpretation: GSP-based suggested labels")
                st.caption(
                    "These are cautious, rule-based interpretive labels derived from the cluster mean profiles. "
                    "They are not additional statistical parameters of the GMM model."
                )
                st.dataframe(
                    interpreted_profile[["intergroup_orientation", "solution_orientation", "final_label"]],
                    use_container_width=True
                )

                with st.expander("Cluster narrative interpretations", expanded=False):
                    for cluster_id, row in interpreted_profile.iterrows():
                        st.markdown(f"**Cluster {cluster_id}: {row['final_label']}**")
                        st.write(row["narrative"])

                # ----------------------------------------------------
                # Cluster profiles by period
                # ----------------------------------------------------
                st.write("Cluster profiles by period")

                profile_period = (
                    community_df
                    .groupby(["period", "cluster"])[features]
                    .mean()
                    .round(3)
                    .reset_index()
                )
                st.dataframe(profile_period, use_container_width=True)

                # ----------------------------------------------------
                # Cluster distribution by period
                # ----------------------------------------------------
                st.write("Cluster distribution by period")

                period_distribution = (
                    community_df
                    .groupby(["period", "cluster"])
                    .size()
                    .reset_index(name="n")
                )

                period_totals = period_distribution.groupby("period")["n"].transform("sum")
                period_distribution["percent"] = (100 * period_distribution["n"] / period_totals).round(1)

                st.dataframe(period_distribution, use_container_width=True)

                fig_bar = px.bar(
                    period_distribution,
                    x="period",
                    y="percent",
                    color=period_distribution["cluster"].astype(str),
                    barmode="stack",
                    title=f"Cluster distribution over time — {community_label}",
                    labels={
                        "period": "Period",
                        "percent": "Percent",
                        "color": "Cluster"
                    }
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # ----------------------------------------------------
                # PCA visualisation
                # ----------------------------------------------------
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X_scaled)

                plot_df = community_df.copy()
                plot_df["PC1"] = coords[:, 0]
                plot_df["PC2"] = coords[:, 1]
                plot_df["cluster"] = plot_df["cluster"].astype(str)
                plot_df["period"] = plot_df["period"].astype(str)

                fig_pca = px.scatter(
                    plot_df,
                    x="PC1",
                    y="PC2",
                    color="cluster",
                    symbol="period",
                    hover_data=features + ["cluster_probability"],
                    title=f"Representational space — {community_label}"
                )
                loadings = pd.DataFrame(
                pca.components_.T,
                columns=["PC1", "PC2"],
                index=features
                )

                st.subheader("PCA loadings")
                st.dataframe(loadings)

                # ----------------------------------------------------
                # Project latest survey respondent into this space
                # ----------------------------------------------------
                latest_response = st.session_state.get("latest_respondent")
                if latest_response is not None:
                    current_scores = build_current_scales(latest_response, historical_df=df)

                    if current_scores.get("community") == comm:
                        missing_current = [
                            feature for feature in features
                            if feature not in current_scores or pd.isna(current_scores.get(feature))
                        ]

                        if missing_current:
                            st.warning(
                                f"Current respondent cannot be projected for {community_label}; "
                                f"missing selected variables: {missing_current}"
                            )
                        else:
                            current_X = pd.DataFrame([{feature: current_scores[feature] for feature in features}])
                            current_scaled = scaler.transform(current_X)
                            current_coords = pca.transform(current_scaled)
                            current_cluster = int(best_model.predict(current_scaled)[0])
                            current_probability = float(best_model.predict_proba(current_scaled).max(axis=1)[0])

                            fig_pca.add_scatter(
                                x=[current_coords[0, 0]],
                                y=[current_coords[0, 1]],
                                mode="markers",
                                name="Current respondent / You are here",
                                marker=dict(size=18, color="black", symbol="star", line=dict(width=2, color="white")),
                                hovertemplate=(
                                    "Current respondent<br>"
                                    "Assigned cluster: %{customdata[0]}<br>"
                                    "Cluster probability: %{customdata[1]:.2f}<br>"
                                    "PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                                ),
                                customdata=[[current_cluster, current_probability]]
                            )

                            st.success(
                                f"Current respondent projected for {community_label}: "
                                f"cluster {current_cluster}, probability {current_probability:.2f}."
                            )

                            # Store latest positioning for this session.
                            st.session_state["current_cluster"] = current_cluster
                            st.session_state["current_cluster_probability"] = current_probability
                            st.session_state["current_scores"] = current_scores
                            st.session_state["current_interpretation"] = interpreted_profile.loc[current_cluster].to_dict()

                            show_current_respondent_position(
                                current_cluster=current_cluster,
                                current_probability=current_probability,
                                current_scores=current_scores,
                                interpreted_profile=interpreted_profile,
                            )

                            with st.expander("Current respondent scale scores", expanded=False):
                                st.dataframe(
                                    pd.DataFrame([{feature: current_scores[feature] for feature in features}]).round(3),
                                    use_container_width=True
                                )

                st.plotly_chart(fig_pca, use_container_width=True)
