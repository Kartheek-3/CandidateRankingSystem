import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyA6wbAFK6-qnXA5VofHj-bnCkplnqG5o0s",
  authDomain: "redrob-ai.firebaseapp.com",
  projectId: "redrob-ai",
  storageBucket: "redrob-ai.firebasestorage.app",
  messagingSenderId: "1017752764147",
  appId: "1:1017752764147:web:266b24d0d4a0a7176a0186",
  measurementId: "G-MLG78SN6TK"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
export const analytics = typeof window !== 'undefined' ? getAnalytics(app) : null;
