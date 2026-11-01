// The page's language is the one Starlight routes it in: Korean at the root,
// English under /en/. Korean strings are the source; English sits beside them.
export const isEn = (lang: string) => lang.startsWith('en');

/** The string for the page's language. */
export const pick = (lang: string, ko: string, en: string) => (isEn(lang) ? en : ko);
