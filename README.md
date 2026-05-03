# DSA Question Bank

A local web app to store and browse DSA questions, solutions, and images.  
Everything is saved as plain files on disk — no database needed.

---

## Requirements

- [Node.js](https://nodejs.org/) **v18 or higher**

Check your version:
```
node -v
```

---

## Setup

**1. Clone or copy this folder to your machine.**

**2. Install dependencies:**
```
npm install
```

That's it. No other setup needed.

---

## Running the app

**Development** (auto-restarts when you edit files):
```
npm run dev
```

**Production** (no auto-restart):
```
npm start
```

Then open your browser at:
```
http://127.0.0.1:8765
```

---

## Data folders

All data is stored inside a `Data/` folder which is **created automatically** the first time you start the server. You don't need to create anything manually.

```
Data/
├── questions.json        ← index of all entries
├── questions/            ← one .txt file per question
├── solutions/            ← one .txt file per solution
└── images/               ← uploaded images
```

Each entry in `questions.json` looks like:
```json
{
  "id": "1",
  "question": "./questions/1.txt",
  "solution": "./solutions/1.txt",
  "images": ["./images/1_1.png"],
  "topic": "Dynamic Programming",
  "subtopic": "Rod Cutting",
  "reference": "CLRS Chapter 15"
}
```

---

## Usage

| Action | How |
|--------|-----|
| Add entry | Fill the form on the left and click **Save** |
| View entries | Entries appear on the right; click **Question** or **Solution** to expand |
| Delete entry | Click the red **Delete** button on any entry |
| Resize panels | Drag the vertical bar in the middle left or right |
| Upload images | Use the Images field; multiple files supported |

---

## Stopping the server

Press `Ctrl + C` in the terminal.
