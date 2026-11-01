// The page's language, as Starlight sets it on <html>. Korean is the source;
// English sits beside it.
export const EN = document.documentElement.lang.startsWith('en');

/** The string for the page's language. */
export const pick = (ko, en) => (EN ? en : ko);
