// src/services/transliterationService.js

/**
 * Simple Transliteration Service using Google Input Tools API
 * Converts Latin text to Devanagari script
 */

const GOOGLE_TRANSLITERATE_API = 'https://inputtools.google.com/request';

class TransliterationService {
  constructor() {
    this.cache = new Map();
  }

  /**
   * Transliterate Latin text to Devanagari
   * @param {string} text - Latin text to transliterate
   * @returns {Promise<string>} - Devanagari text
   */
  async transliterate(text) {
    if (!text || text.trim() === '') {
      return '';
    }

    // Check cache first
    const cacheKey = text.toLowerCase().trim();
    if (this.cache.has(cacheKey)) {
      console.log('Using cached transliteration for:', text);
      return this.cache.get(cacheKey);
    }

    // Try Sanskrit first, then Hindi as fallback
    const languageCodes = ['sa-t-i0-und', 'hi-t-i0-und'];
    
    for (const itc of languageCodes) {
      try {
        console.log(`Attempting transliteration with language code: ${itc}`);
        
        const response = await fetch(GOOGLE_TRANSLITERATE_API, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            method: 'transliteration',
            apikey: 'AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw',
            params: {
              text: text,
              itc: itc,
              num: 5,
            }
          })
        });

        if (!response.ok) {
          console.warn(`API request failed with ${itc}:`, response.status);
          continue;
        }

        const data = await response.json();
        console.log('API Response:', data);
        
        if (data && data[1] && data[1][0] && data[1][0][1] && data[1][0][1].length > 0) {
          const transliterated = data[1][0][1][0];
          console.log('Transliteration successful:', text, '→', transliterated);
          this.cache.set(cacheKey, transliterated);
          return transliterated;
        }
      } catch (error) {
        console.error(`Error with ${itc}:`, error);
        continue;
      }
    }

    // If all attempts fail, try manual mapping for common Brahmi characters
    console.warn('API transliteration failed, trying manual mapping');
    const manualTranslation = this.manualTransliterate(text);
    if (manualTranslation !== text) {
      this.cache.set(cacheKey, manualTranslation);
      return manualTranslation;
    }

    console.warn('No transliteration found, returning original text');
    return text; // Return original if all methods fail
  }

  /**
   * Manual transliteration mapping for common Brahmi characters
   * @param {string} text - Latin text
   * @returns {string} - Devanagari text
   */
  manualTransliterate(text) {
    const mapping = {
      // ---------- VOWEL KEYS (independent vowels) ----------
      'a': 'अ', 'aa': 'आ', 'i': 'इ', 'ii': 'ई', 'u': 'उ', 'uu': 'ऊ',
      'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ',
      'ri_vowel': 'ऋ', 'rri_vowel': 'ॠ', 'li_vowel': 'ऌ', 'lli_vowel': 'ॡ',
      'am': 'अं', 'ah': 'अः', 'an': 'अँ',

      // ---------- VELAR ----------
      'ka': 'क', 'kaa': 'का', 'ki': 'कि', 'kii': 'की', 'ku': 'कु', 'kuu': 'कू',
      'ke': 'के', 'kai': 'कै', 'ko': 'को', 'kau': 'कौ', 'kri': 'कृ', 'k_': 'क्',

      'kha': 'ख', 'khaa': 'खा', 'khi': 'खि', 'khii': 'खी', 'khu': 'खु', 'khuu': 'खू',
      'khe': 'खे', 'khai': 'खै', 'kho': 'खो', 'khau': 'खौ', 'khri': 'खृ', 'kh_': 'ख्',

      'ga': 'ग', 'gaa': 'गा', 'gi': 'गि', 'gii': 'गी', 'gu': 'गु', 'guu': 'गू',
      'ge': 'गे', 'gai': 'गै', 'go': 'गो', 'gau': 'गौ', 'gri': 'गृ', 'g_': 'ग्',

      'gha': 'घ', 'ghaa': 'घा', 'ghi': 'घि', 'ghii': 'घी', 'ghu': 'घु', 'ghuu': 'घू',
      'ghe': 'घे', 'ghai': 'घै', 'gho': 'घो', 'ghau': 'घौ', 'ghri': 'घृ', 'gh_': 'घ्',

      'nga': 'ङ', 'ngaa': 'ङा', 'ngi': 'ङि', 'ngii': 'ङी', 'ngu': 'ङु', 'nguu': 'ङू',
      'nge': 'ङे', 'ngai': 'ङै', 'ngo': 'ङो', 'ngau': 'ङौ', 'ngri': 'ङृ', 'ng_': 'ङ्',

      // ---------- PALATAL ----------
      'ca': 'च', 'caa': 'चा', 'ci': 'चि', 'cii': 'ची', 'cu': 'चु', 'cuu': 'चू',
      'ce': 'चे', 'cai': 'चै', 'co': 'चो', 'cau': 'चौ', 'cri': 'चृ', 'c_': 'च्',

      'cha': 'छ', 'chaa': 'छा', 'chi': 'छि', 'chii': 'छी', 'chu': 'छु', 'chuu': 'छू',
      'che': 'छे', 'chai': 'छै', 'cho': 'छो', 'chau': 'छौ', 'chri': 'छृ', 'ch_': 'छ्',

      'ja': 'ज', 'jaa': 'जा', 'ji': 'जि', 'jii': 'जी', 'ju': 'जु', 'juu': 'जू',
      'je': 'जे', 'jai': 'जै', 'jo': 'जो', 'jau': 'जौ', 'jri': 'जृ', 'j_': 'ज्',

      'jha': 'झ', 'jhaa': 'झा', 'jhi': 'झि', 'jhii': 'झी', 'jhu': 'झु', 'jhuu': 'झू',
      'jhe': 'झे', 'jhai': 'झै', 'jho': 'झो', 'jhau': 'झौ', 'jhri': 'झृ', 'jh_': 'झ्',

      'nya': 'ञ', 'nyaa': 'ञा', 'nyi': 'ञि', 'nyii': 'ञी', 'nyu': 'ञु', 'nyuu': 'ञू',
      'nye': 'ञे', 'nyai': 'ञै', 'nyo': 'ञो', 'nyau': 'ञौ', 'nyri': 'ञृ', 'ny_': 'ञ्',

      // ---------- RETROFLEX (CEREBRAL) ----------
      'tta': 'ट', 'ttaa': 'टा', 'tti': 'टि', 'ttii': 'टी', 'ttu': 'टु', 'ttuu': 'टू',
      'tte': 'टे', 'ttai': 'टै', 'tto': 'टो', 'ttau': 'टौ', 'ttri': 'टृ', 'tt_': 'ट्',

      'ttha': 'ठ', 'tthaa': 'ठा', 'tthi': 'ठि', 'tthii': 'ठी', 'tthu': 'ठु', 'tthuu': 'ठू',
      'tthe': 'ठे', 'tthai': 'ठै', 'ttho': 'ठो', 'tthau': 'ठौ', 'tthri': 'ठृ', 'tth_': 'ठ्',

      'dda': 'ड', 'ddaa': 'डा', 'ddi': 'डि', 'ddii': 'डी', 'ddu': 'डु', 'dduu': 'डू',
      'dde': 'डे', 'ddai': 'डै', 'ddo': 'डो', 'ddau': 'डौ', 'ddri': 'डृ', 'dd_': 'ड्',

      'ddha': 'ढ', 'ddhaa': 'ढा', 'ddhi': 'ढि', 'ddhii': 'ढी', 'ddhu': 'ढु', 'ddhuu': 'ढू',
      'ddhe': 'ढे', 'ddhai': 'ढै', 'ddho': 'ढो', 'ddhau': 'ढौ', 'ddhri': 'ढृ', 'ddh_': 'ढ्',

      'nna': 'ण', 'nnaa': 'णा', 'nni': 'णि', 'nnii': 'णी', 'nnu': 'णु', 'nnuu': 'णू',
      'nne': 'णे', 'nnai': 'णै', 'nno': 'णो', 'nnau': 'णौ', 'nnri': 'णृ', 'nn_': 'ण्',

      // ---------- DENTAL ----------
      'ta': 'त', 'taa': 'ता', 'ti': 'ति', 'tii': 'ती', 'tu': 'तु', 'tuu': 'तू',
      'te': 'ते', 'tai': 'तै', 'to': 'तो', 'tau': 'तौ', 'tri': 'तृ', 't_': 'त्',

      'tha': 'थ', 'thaa': 'था', 'thi': 'थि', 'thii': 'थी', 'thu': 'थु', 'thuu': 'थू',
      'the': 'थे', 'thai': 'थै', 'tho': 'थो', 'thau': 'थौ', 'thri': 'थृ', 'th_': 'थ्',

      'da': 'द', 'daa': 'दा', 'di': 'दि', 'dii': 'दी', 'du': 'दु', 'duu': 'दू',
      'de': 'दे', 'dai': 'दै', 'do': 'दो', 'dau': 'दौ', 'dri': 'दृ', 'd_': 'द्',

      'dha': 'ध', 'dhaa': 'धा', 'dhi': 'धि', 'dhii': 'धी', 'dhu': 'धु', 'dhuu': 'धू',
      'dhe': 'धे', 'dhai': 'धै', 'dho': 'धो', 'dhau': 'धौ', 'dhri': 'धृ', 'dh_': 'ध्',

      'na': 'न', 'naa': 'ना', 'ni': 'नि', 'nii': 'नी', 'nu': 'नु', 'nuu': 'नू',
      'ne': 'ने', 'nai': 'नै', 'no': 'नो', 'nau': 'नौ', 'nri': 'नृ', 'n_': 'न्',

      // ---------- LABIAL ----------
      'pa': 'प', 'paa': 'पा', 'pi': 'पि', 'pii': 'पी', 'pu': 'पु', 'puu': 'पू',
      'pe': 'पे', 'pai': 'पै', 'po': 'पो', 'pau': 'पौ', 'pri': 'पृ', 'p_': 'प्',

      'pha': 'फ', 'phaa': 'फा', 'phi': 'फि', 'phii': 'फी', 'phu': 'फु', 'phuu': 'फू',
      'phe': 'फे', 'phai': 'फै', 'pho': 'फो', 'phau': 'फौ', 'phri': 'फृ', 'ph_': 'फ्',

      'ba': 'ब', 'baa': 'बा', 'bi': 'बि', 'bii': 'बी', 'bu': 'बु', 'buu': 'बू',
      'be': 'बे', 'bai': 'बै', 'bo': 'बो', 'bau': 'बौ', 'bri': 'बृ', 'b_': 'ब्',

      'bha': 'भ', 'bhaa': 'भा', 'bhi': 'भि', 'bhii': 'भी', 'bhu': 'भु', 'bhuu': 'भू',
      'bhe': 'भे', 'bhai': 'भै', 'bho': 'भो', 'bhau': 'भौ', 'bhri': 'भृ', 'bh_': 'भ्',

      'ma': 'म', 'maa': 'मा', 'mi': 'मि', 'mii': 'मी', 'mu': 'मु', 'muu': 'मू',
      'me': 'मे', 'mai': 'मै', 'mo': 'मो', 'mau': 'मौ', 'mri': 'मृ', 'm_': 'म्',

      // ---------- SEMI-VOWELS / APPROXIMANTS ----------
      'ya': 'य', 'yaa': 'या', 'yi': 'यि', 'yii': 'यी', 'yu': 'यु', 'yuu': 'यू',
      'ye': 'ये', 'yai': 'यै', 'yo': 'यो', 'yau': 'यौ', 'yri': 'यृ', 'y_': 'य्',

      'ra': 'र', 'raa': 'रा', 'ri': 'रि', 'rii': 'री', 'ru': 'रु', 'ruu': 'रू',
      're': 'रे', 'rai': 'रै', 'ro': 'रो', 'rau': 'रौ', 'rri': 'रृ', 'r_': 'र्',

      'la': 'ल', 'laa': 'ला', 'li': 'लि', 'lii': 'ली', 'lu': 'लु', 'luu': 'लू',
      'le': 'ले', 'lai': 'लै', 'lo': 'लो', 'lau': 'लौ', 'lri': 'लृ', 'l_': 'ल्',

      'va': 'व', 'vaa': 'वा', 'vi': 'वि', 'vii': 'वी', 'vu': 'वु', 'vuu': 'वू',
      've': 'वे', 'vai': 'वै', 'vo': 'वो', 'vau': 'वौ', 'vri': 'वृ', 'v_': 'व्',

      // ---------- SIBILANTS & ASPIRATE ----------
      'sha': 'श', 'shaa': 'शा', 'shi': 'शि', 'shii': 'शी', 'shu': 'शु', 'shuu': 'शू',
      'she': 'शे', 'shai': 'शै', 'sho': 'शो', 'shau': 'शौ', 'shri': 'शृ', 'sh_': 'श्',

      'ssa': 'ष', 'ssaa': 'षा', 'ssi': 'षि', 'ssii': 'षी', 'ssu': 'षु', 'ssuu': 'षू',
      'sse': 'षे', 'ssai': 'षै', 'sso': 'षो', 'ssau': 'षौ', 'ssri': 'षृ', 'ss_': 'ष्',

      'sa': 'स', 'saa': 'सा', 'si': 'सि', 'sii': 'सी', 'su': 'सु', 'suu': 'सू',
      'se': 'से', 'sai': 'सै', 'so': 'सो', 'sau': 'सौ', 'sri': 'सृ', 's_': 'स्',

      'ha': 'ह', 'haa': 'हा', 'hi': 'हि', 'hii': 'ही', 'hu': 'हु', 'huu': 'हू',
      'he': 'हे', 'hai': 'है', 'ho': 'हो', 'hau': 'हौ', 'hri': 'हृ', 'h_': 'ह्',

      // ---------- MARATHI / ADDITIONAL CONSONANTS ----------
      'lla': 'ळ', 'llaa': 'ळा', 'lli': 'ळि', 'llii': 'ळी', 'llu': 'ळु', 'lluu': 'ळू',
      'lle': 'ळे', 'llai': 'ळै', 'llo': 'ळो', 'llau': 'ळौ', 'llri': 'ळृ', 'll_': 'ळ्',

      'lra': 'ऴ', 'lraa': 'ऴा', 'lri': 'ऴि', 'lrii': 'ऴी', 'lru': 'ऴु', 'lruu': 'ऴू',
      'lre': 'ऴे', 'lrai': 'ऴै', 'lro': 'ऴो', 'lrau': 'ऴौ', 'lrri': 'ऴृ', 'lr_': 'ऴ्',

      'rra': 'ऱ', 'rraa': 'ऱा', 'rri_': 'ऱि', 'rrii': 'ऱी', 'rru': 'ऱु', 'rruu': 'ऱू',
      'rre': 'ऱे', 'rrai': 'ऱै', 'rro': 'ऱो', 'rrau': 'ऱौ', 'rrri': 'ऱृ', 'rr_': 'ऱ्',

      // ---------- COMMON COMPOUNDS (explicit) ----------
      'ksha': 'क्ष', 'kshaa': 'क्षा', 'kshi': 'क्षि', 'kshii': 'क्षी', 'kshu': 'क्षु', 'kshuu': 'क्षू',
      'kshe': 'क्षे', 'kshai': 'क्षै', 'ksho': 'क्षो', 'kshau': 'क्षौ', 'kshr': 'क्ष्र', 'ksh_': 'क्ष्',

      'jna': 'ज्ञ', 'jnaa': 'ज्ञा', 'jni': 'ज्ञि', 'jnii': 'ज्ञी', 'jnu': 'ज्ञु', 'jnuu': 'ज्ञू',
      'jne': 'ज्ञे', 'jnai': 'ज्ञै', 'jno': 'ज्ञो', 'jnau': 'ज्ञौ', 'jnri': 'ज्ञृ', 'jn_': 'ज्ञ्',

      'tra': 'त्र', 'traa': 'त्रा', 'tri': 'त्रि', 'trii': 'त्री', 'tru': 'त्रु', 'truu': 'त्रू',
      'tre': 'त्रे', 'trai': 'त्रै', 'tro': 'त्रो', 'trau': 'त्रौ', 'trri': 'त्रृ', 'tr_': 'त्र्',

      // ---------- NUKTA (Perso-Arabic influenced) ----------
      'qa': 'क़', 'qaa': 'क़ा', 'qi': 'क़ि', 'qii': 'क़ी', 'qu': 'क़ु', 'quu': 'क़ू',
      'qe': 'क़े', 'qai': 'क़ै', 'qo': 'क़ो', 'qau': 'क़ौ', 'qr': 'क़ृ', 'q_': 'क़्',

      'za': 'ज़', 'zaa': 'ज़ा', 'zi': 'ज़ि', 'zii': 'ज़ी', 'zu': 'ज़ु', 'zuu': 'ज़ू',
      'ze': 'ज़े', 'zai': 'ज़ै', 'zo': 'ज़ो', 'zau': 'ज़ौ', 'zr': 'ज़ृ', 'z_': 'ज़्',

      'fa': 'फ़', 'faa': 'फ़ा', 'fi': 'फ़ि', 'fii': 'फ़ी', 'fu': 'फ़ु', 'fuu': 'फ़ू',
      'fe': 'फ़े', 'fai': 'फ़ै', 'fo': 'फ़ो', 'fau': 'फ़ौ', 'fr': 'फ़ृ', 'f_': 'फ़्',

      'rda': 'ड़', 'rdaa': 'ड़ा', 'rdi': 'ड़ि', 'rdii': 'ड़ी', 'rdu': 'ड़ु', 'rduu': 'ड़ू',
      'rde': 'ड़े', 'rdai': 'ड़ै', 'rdo': 'ड़ो', 'rdau': 'ड़ौ', 'rdr': 'ड़ृ', 'rd_': 'ड़्',

      'rdha': 'ढ़', 'rdhaa': 'ढ़ा', 'rdhi': 'ढ़ि', 'rdhii': 'ढ़ी', 'rdhu': 'ढ़ु', 'rdhuu': 'ढ़ू',
      'rdhe': 'ढ़े', 'rdhai': 'ढ़ै', 'rdho': 'ढ़ो', 'rdhau': 'ढ़ौ', 'rdhr': 'ढ़ृ', 'rdh_': 'ढ़्',

      // ---------- PUNCTUATION / DIACRITICS ----------
      'anusvara': 'ं', 'visarga': 'ः', 'chandrabindu': 'ँ', 'virama': '्',
    };

    const lowerText = text.toLowerCase().trim();
    
    // Try exact match first
    if (mapping[lowerText]) {
      return mapping[lowerText];
    }

    // Try splitting by spaces and transliterating each word
    const words = lowerText.split(/\s+/);
    if (words.length > 1) {
      return words.map(word => mapping[word] || word).join(' ');
    }

    // Try finding longest matching substring
    for (let len = lowerText.length; len > 0; len--) {
      for (let i = 0; i <= lowerText.length - len; i++) {
        const substr = lowerText.substring(i, i + len);
        if (mapping[substr]) {
          return mapping[substr];
        }
      }
    }

    return text;
  }

  /**
   * Clear the cache
   */
  clearCache() {
    this.cache.clear();
  }
}

// Create and export singleton instance
const transliterationService = new TransliterationService();
export default transliterationService;