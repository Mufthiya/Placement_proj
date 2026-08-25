const firebaseConfig = {
  apiKey: "AIzaSyA6v3RA1tGpaj3ul-tud7XVdVlViH8nP4Q",
  authDomain: "placement-proj.firebaseapp.com",
  projectId: "placement-proj",
  storageBucket: "placement-proj.firebasestorage.app",
  messagingSenderId: "891768048273",
  appId: "1:891768048273:web:f307dbb4d4254efc9e8d30",
  measurementId: "G-V1RXVMDGX1",
};

function fbError(e) {
  return (e && e.message)
    ? e.message.replace("Firebase: ", "").replace(/\(auth.*\)\.?/, "").trim()
    : "Something went wrong";
}

if (typeof firebase === "undefined") {
  console.error("Firebase SDK did not load - check your internet connection.");
  document.addEventListener("DOMContentLoaded", () => {
    const msg = document.getElementById("authMsg");
    if (msg) msg.textContent = "Firebase SDK failed to load - check your internet connection.";
  });
} else {

  if (!firebase.apps.length) {
    firebase.initializeApp(firebaseConfig);
  }
  try {
    firebase.analytics();
  } catch (e) { /* analytics unavailable offline */ }

  const fbAuth = firebase.auth();
  const fbDb = firebase.firestore();

  async function saveUserDoc(user, provider) {
    const ref = fbDb.collection("users").doc(user.uid);
    const doc = await ref.get();
    const base = {
      name: user.displayName || user.email.split("@")[0],
      email: user.email,
      provider: provider,
      photoURL: user.photoURL || "",
      lastLoginAt: firebase.firestore.FieldValue.serverTimestamp(),
    };
    if (!doc.exists) {
      base.createdAt = firebase.firestore.FieldValue.serverTimestamp();
      await ref.set(base);
    } else {
      await ref.set(base, { merge: true });
    }
  }

  fbAuth.onAuthStateChanged(async (user) => {
    document.querySelectorAll("[data-auth='guest']").forEach((el) => {
      el.style.display = user ? "none" : "";
    });
    document.querySelectorAll("[data-auth='user']").forEach((el) => {
      el.style.display = user ? "" : "none";
    });

    if (user) {
      const label = user.displayName || user.email;
      document.querySelectorAll("[data-user-name]").forEach((el) => {
        el.textContent = label;
      });
      const nameInput = document.getElementById("f_name");
      if (nameInput && !nameInput.value && user.displayName) {
        nameInput.value = user.displayName;
      }

      try {
        const snap = await fbDb.collection("users").doc(user.uid).get();
        const d = snap.data() || {};
        const created = d.createdAt ? d.createdAt.toDate().toDateString() : "today";
        const info = document.getElementById("authInfo");
        if (info) {
          info.innerHTML =
            "<div class='rc-title'>Signed in as</div>" +
            "<div class='rc-text'>" + (d.name || label) + "</div>" +
            "<p class='small'>" + user.email + "<br>Provider: " +
            (d.provider || "password") + "<br>Member since: " + created +
            "<br>UID: " + user.uid.slice(0, 12) + "...</p>";
        }
        const myName = document.getElementById("my_name");
        if (myName) myName.textContent = d.name || label;
      } catch (e) {
        console.warn("Firestore read failed:", e.message);
      }
    }
  });

  async function fbSignup(name, email, password) {
    const cred = await fbAuth.createUserWithEmailAndPassword(email, password);
    if (name) await cred.user.updateProfile({ displayName: name });
    await saveUserDoc(cred.user, "password");
    window.location.href = "/dash";
  }

  async function fbSignin(email, password) {
    const cred = await fbAuth.signInWithEmailAndPassword(email, password);
    await saveUserDoc(cred.user, "password");
    window.location.href = "/dash";
  }

  async function fbGoogle() {
    const provider = new firebase.auth.GoogleAuthProvider();
    const cred = await fbAuth.signInWithPopup(provider);
    await saveUserDoc(cred.user, "google.com");
    window.location.href = "/dash";
  }

  async function fbLogout() {
    await fbAuth.signOut();
    window.location.href = "/";
  }

  window.fbSignup = fbSignup;
  window.fbSignin = fbSignin;

  document.addEventListener("DOMContentLoaded", () => {
    const tabIn = document.getElementById("tabSignin");
    const tabUp = document.getElementById("tabSignup");
    if (tabIn && tabUp) {
      const formIn = document.getElementById("formSignin");
      const formUp = document.getElementById("formSignup");
      const pick = (mode) => {
        const isIn = mode === "in";
        formIn.classList.toggle("hidden", !isIn);
        formUp.classList.toggle("hidden", isIn);
        tabIn.classList.toggle("active-tab", isIn);
        tabUp.classList.toggle("active-tab", !isIn);
      };
      tabIn.addEventListener("click", () => pick("in"));
      tabUp.addEventListener("click", () => pick("up"));
    }

    const sIn = document.getElementById("doSignin");
    if (sIn) sIn.addEventListener("click", async () => {
      const msg = document.getElementById("authMsg");
      try {
        sIn.disabled = true;
        await fbSignin(document.getElementById("si_email").value.trim(),
                       document.getElementById("si_pass").value);
      } catch (e) {
        msg.textContent = fbError(e);
        sIn.disabled = false;
      }
    });

    const sUp = document.getElementById("doSignup");
    if (sUp) sUp.addEventListener("click", async () => {
      const msg = document.getElementById("authMsg");
      try {
        sUp.disabled = true;
        const pass = document.getElementById("su_pass").value;
        if (pass.length < 6) throw new Error("Password must be at least 6 characters");
        await fbSignup(document.getElementById("su_name").value.trim(),
                       document.getElementById("su_email").value.trim(), pass);
      } catch (e) {
        msg.textContent = fbError(e);
        sUp.disabled = false;
      }
    });

    const gBtn = document.getElementById("doGoogle");
    if (gBtn) gBtn.addEventListener("click", () => fbGoogle().catch((e) => {
      document.getElementById("authMsg").textContent = fbError(e);
    }));

    document.querySelectorAll("[data-logout]").forEach((btn) => {
      btn.addEventListener("click", () => fbLogout());
    });
  });
}
