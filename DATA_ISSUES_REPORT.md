# דוח בעיות בנתונים - ML Dataset

## 🔴 בעיות קריטיות

### 1. דליפת נתונים (Data Leakage) - **הבעיה החמורה ביותר!**

| חפיפה | כמות | אחוז |
|--------|------|------|
| Train ∩ Test | 7,140 גנים | 85.7% מה-Test! |
| Train ∩ Validation | 4,577 גנים | 100% מה-Validation! |
| Train ∩ Test (רצפים) | 7,204 רצפים | |

**המשמעות**: ה-Validation כולו מכיל גנים שנמצאים גם ב-Train! המודל "יכיר" את הדוגמאות מראש, מה שיוביל לתוצאות אופטימיות מזויפות.

**פתרונות:**
- ליצור חלוקה חדשה לגמרי עם סטרטיפיקציה
- לוודא שאין חפיפה ברמת ה-NCBIGeneID או הרצף
- להשתמש ב-GroupKFold לפי משפחות גנים

---

### 2. חוסר איזון קיצוני בקלאסים (Class Imbalance)

| קלאס | כמות | אחוז |
|------|------|------|
| PSEUDO | 10,220 | 45.2% |
| BIOLOGICAL_REGION | 6,925 | 30.7% |
| ncRNA | 2,497 | 11.1% |
| snoRNA | 1,148 | 5.1% |
| PROTEIN_CODING | 524 | 2.3% |
| tRNA | 488 | 2.2% |
| OTHER | 366 | 1.6% |
| rRNA | 277 | 1.2% |
| snRNA | 145 | 0.6% |
| **scRNA** | **3** | **0.01%** |

**יחס חוסר איזון: 3,407:1** (בין PSEUDO ל-scRNA)

**פתרונות:**
```python
# 1. שימוש ב-Class Weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y_train)

# 2. Oversampling לקלאסים נדירים
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X, y)

# 3. Undersampling לקלאסים גדולים
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy={
    'PSEUDO': 2000,
    'BIOLOGICAL_REGION': 2000
})
```

---

### 3. קלאס scRNA חסר ב-Validation

**הבעיה:** 0 דגימות של scRNA ב-Validation set

**פתרונות:**
1. **מיזוג קלאסים:** לאחד scRNA, snRNA, snoRNA לקטגוריה "small_RNA"
2. **הסרת הקלאס:** להסיר את scRNA לגמרי (רק 4 דגימות)
3. **העברת דגימה:** להעביר דגימה אחת מ-Train/Test ל-Validation

---

## 🟡 בעיות בינוניות

### 4. פיצ'ר קבוע (Constant Feature)

`GeneGroupMethod` = "NCBI Ortholog" בכל הדגימות

**פתרון:** להסיר את העמודה - לא מוסיפה מידע

```python
df = df.drop(columns=['GeneGroupMethod'])
```

---

### 5. רצפים כפולים (Duplicate Sequences)

| סט | רצפים כפולים |
|----|--------------|
| Train | 709 |
| Test | 120 |
| Validation | 0 |

**פתרון:** לבדוק האם אלו באמת כפילויות או גנים שונים עם רצפים זהים

---

### 6. שונות גבוהה באורכי הרצפים

| סטטיסטיקה | ערך |
|-----------|-----|
| מינימום | 2 |
| מקסימום | 1,000 |
| ממוצע | 360 |
| חציון | 295 |
| סטיית תקן | 260 |

**התפלגות אורכים:**
- 0-50: 113 (0.5%)
- 51-100: 3,356 (14.9%)
- 101-200: 3,814 (16.9%)
- 201-500: 9,359 (41.4%)
- 501-1000: 5,951 (26.3%)

**אורך ממוצע לפי סוג גן:**
| סוג גן | אורך ממוצע |
|--------|-----------|
| PROTEIN_CODING | 742 |
| PSEUDO | 436 |
| OTHER | 338 |
| BIOLOGICAL_REGION | 328 |
| ncRNA | 266 |
| scRNA | 219 |
| rRNA | 144 |
| snRNA | 126 |
| snoRNA | 111 |
| tRNA | 75 |

**פתרונות:**
```python
# 1. Padding לאורך קבוע
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 500
X_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# 2. שימוש באורך כפיצ'ר נוסף
df['seq_length'] = df['NucleotideSequence'].str.len()

# 3. נרמול לפי אורך
# מתאים לפיצ'רים כמו GC content
```

---

## 🟢 ממצאים חיוביים

✅ **אין ערכים חסרים** - כל העמודות מלאות  
✅ **אין שורות כפולות לגמרי**  
✅ **כל הרצפים תקינים** - מכילים רק ACGTU  

---

## המלצות לפעולה

### שלב 1: תיקון דליפת הנתונים (קריטי!)

```python
from sklearn.model_selection import train_test_split

# טעינת כל הנתונים
all_data = pd.concat([train, test, validation])

# הסרת כפילויות
all_data = all_data.drop_duplicates(subset=['NCBIGeneID'])

# חלוקה חדשה עם stratified split
train_new, temp = train_test_split(
    all_data, 
    test_size=0.35, 
    stratify=all_data['GeneType'],
    random_state=42
)

test_new, val_new = train_test_split(
    temp, 
    test_size=0.37,  # ~13% מהכל
    stratify=temp['GeneType'],
    random_state=42
)
```

### שלב 2: טיפול בחוסר איזון

```python
# אופציה 1: מיזוג קלאסים נדירים
def merge_rare_classes(gene_type):
    rare_rna = ['scRNA', 'snRNA', 'rRNA']
    if gene_type in rare_rna:
        return 'rare_RNA'
    return gene_type

df['GeneType'] = df['GeneType'].apply(merge_rare_classes)

# אופציה 2: Focal Loss
import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = self.alpha * tf.pow(1 - p_t, self.gamma)
        return focal_weight * ce
```

### שלב 3: עיבוד הרצפים

```python
# המרת רצפים ל-K-mers
def sequence_to_kmers(seq, k=3):
    seq = seq.strip('<>')
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    return ' '.join(kmers)

df['kmers'] = df['NucleotideSequence'].apply(lambda x: sequence_to_kmers(x, k=3))

# One-hot encoding
def one_hot_encode(seq, max_len=500):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    seq = seq.strip('<>')[:max_len]
    encoded = np.zeros((max_len, 4))
    for i, char in enumerate(seq):
        if char in mapping:
            encoded[i, mapping[char]] = 1
    return encoded
```

---

## סיכום עדיפויות

| עדיפות | בעיה | פעולה |
|--------|------|-------|
| 1️⃣ | דליפת נתונים | ליצור חלוקה חדשה |
| 2️⃣ | קלאס חסר ב-Validation | למזג/להסיר scRNA |
| 3️⃣ | חוסר איזון | class weights / oversampling |
| 4️⃣ | פיצ'ר קבוע | להסיר GeneGroupMethod |
| 5️⃣ | אורכי רצפים | padding/truncation |

---

*נוצר על ידי data_issues_analysis.py*
