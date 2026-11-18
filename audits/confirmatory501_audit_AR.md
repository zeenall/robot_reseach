# تقرير التدقيق العلمي – Paper 2، تجربة 501 حالة مستقلة

**تاريخ المراجعة:** 2026-11-11  
**المدخل:** `results (30).zip`، المجلد `PAPER2_CONFIRM_501`  
**نوع التجربة:** تقييم تأكيدي مجمّد للأوزان، دون تدريب أو اختيار معاملات على الاختبار.

## 1. بروتوكول التجربة

- 501 حالة: 167 One-hot، و167 Multi-hot، و167 Hard.
- النماذج: `stage_a`، و`uniform_kl_20261112`، و`attention_original`.
- لكل نموذج: K=1 من Guided Greedy، وK=4 من أربعة مرشحين: Guided Greedy، Guided Sample، Scarcity Greedy، Scarcity Sample.
- معيار التقييم المسبق الرئيسي: Penalized Makespan في K=4؛ المقارنة الأساسية المسبقة: **Uniform+KL ناقص AttentionNet**.
- مقياس العقوبة للحالة غير المكتملة 200، لكن كل النتائج الفائزة مكتملة في هذا الاختبار؛ لذا يساوي Penalized Makespan الـMakespan.
- المقارنة الإحصائية: فروق زوجية مع Bootstrap طبقي بحسب نوع الحالة (20,000 عينة، فترة ثقة مئينية 95%).
- جميع الأوزان ثابتة، وتطابقت SHA-256 للنماذج الثلاثة مع `locked_protocol.json`.

## 2. أهم المتوسطات

| النموذج | K=1 Makespan | K=4 Makespan | K=4 Waiting | K=4 Search (ثانية) | النجاح |
|---|---:|---:|---:|---:|---:|
| Stage A | 27.1452 | **25.5974** | 4.4934 | **1.4948** | 501/501 |
| Uniform + KL (20261112) | 27.1799 | 25.6680 | 4.4950 | 1.4921 | 501/501 |
| AttentionNet (original weights) | **26.9419** | 25.6704 | **4.2640** | 2.3622 | 501/501 |

عند K=4، Stage A أسرع من AttentionNet بنحو **36.7% من زمن AttentionNet**، لكن أفضلية Makespan البالغة 0.0730 ليست ذات دلالة إحصائية. الفرق الرئيسي Uniform+KL مقابل AttentionNet يساوي **−0.0024**، وبذلك لا يثبت تفوق Uniform.

## 3. الفروق الزوجية والثقة 95%

| المقارنة | K | متوسط الفرق | فترة الثقة 95% |
|---|---:|---:|---|
| Uniform + KL − AttentionNet (أساسية محددة مسبقاً) | 4 | −0.0024 | [−0.1574, +0.1497] |
| Stage A − AttentionNet (ثانوية) | 4 | −0.0730 | [−0.2302, +0.0842] |
| Uniform + KL − Stage A (ثانوية) | 4 | +0.0706 | [−0.0316, +0.1741] |
| Uniform + KL − AttentionNet | 1 | +0.2380 | [+0.0250, +0.4509] |
| Stage A − AttentionNet | 1 | +0.2033 | [−0.0210, +0.4293] |

*الفارق السالب لمصلحة النموذج الأول؛ جميع فترات الثقة هنا حسب الملف الأصلي. أعدت حساب فترات الثقة الأساسية بصورة مستقلة ببذرة أخرى وأعطت النتائج نفسها تقريباً مع فروق مونت كارلو طفيفة؛ راجع `independent_bootstrap_check.csv`.*

## 4. تحليل الفئات عند K=4

| الفئة | Stage A | Uniform + KL | AttentionNet |
|---|---:|---:|---:|
| One-hot | 28.0996 | 28.2512 | **27.9877** |
| Multi-hot | 20.6975 | **20.6117** | 21.0251 |
| Hard | **27.9952** | 28.1412 | 27.9984 |

- **One-hot:** AttentionNet أفضل عددياً. Uniform − AttentionNet = +0.2635، CI [−0.0151, +0.5470] (غير حاسم).
- **Multi-hot:** Uniform − AttentionNet = −0.4134، CI [−0.6658, −0.1613]؛ Stage A − AttentionNet = −0.3276، CI [−0.5933, −0.0666]. هذا تحسن استكشافي ضمن الفئة، مع التحفظ بشأن تعدد المقارنات.
- **Hard:** Stage A شبه مطابق عددياً لـAttentionNet؛ Uniform يتراجع قليلاً.

## 5. هل علّمنا Distillation سياسة أفضل؟

**ليس بصورة عامة.** عند K=4، Uniform+KL أضعف من Stage A بنحو 0.0706 وبفترة ثقة تشمل الصفر؛ وعند K=1 هو أضعف عددياً من Stage A وAttentionNet. لذا لا يصح إعلان أن تدريب Uniform+KL رفع الجودة العامة في هذا الاختبار. بالمقابل، يتضح أن تحسين **البحث** من K=1 إلى K=4 مهم:

| النموذج | انخفاض Makespan من K=1 إلى K=4 | النسبة |
|---|---:|---:|
| Stage A | 1.5477 | 5.70% |
| Uniform + KL | 1.5119 | 5.56% |
| AttentionNet | 1.2715 | 4.72% |

## 6. سلامة الملفات

- 3006 صف تقييم = 501 حالة × 3 نماذج × ميزانيتَي بحث.
- جميع الصفوف `complete=1` و`replay_verified=1`، بدون قيم مفقودة أو مفاتيح تقييم مكررة.
- لكل صف K=4 بالضبط أربع محاولات، ولكل صف K=1 محاولة واحدة. مرشح واحد من 6012 مرشح K=4 لم يكمل، لكن الخطة الفائزة أكملت بنجاح.
- تم فحص تجميع الحالات، بصمات الأوزان ونتائج الاختبارات، وأعدت حساب المتوسطات وفترات الثقة بشكل مستقل. **التحقق المستقل هنا هو تدقيق للسجلات؛ لم أعد تشغيل الـ3006 جدول داخل المحاكي من الصفر.**

## 7. القيد المنهجي الأساسي

هذه المقارنة تستخدم أوزان AttentionNet المنشورة في **بروتوكول استدلال مشترك معدّل** (Guided + Scarcity)، ولذلك لا تعني إعادة إنتاج جميع مكونات خوارزمية الباحثين كما وردت في ورقتهم الأصلية. والنتائج تقيس تخطيطاً مسبقاً بمساعدة المحاكي، وليس إثباتاً لجدولة Online تحت أحداث غير متوقعة.

النتيجة الأساسية المحددة سلفاً لم تحقق تفوقاً إحصائياً؛ لا ينبغي تغيير النموذج الرئيسي بعد الاطلاع على نتائج الـ501 ثم تسميتها نتائج تأكيدية جديدة. كما أن عدم دلالة الفارق لا يثبت التكافؤ؛ يجب وضع هامش تكافؤ/عدم دونية مسبقاً إذا كان هذا هو الادعاء المستهدف.

## 8. ما يصلح كتابته في البحث

> On 501 previously unseen instances with frozen policies and a fixed four-rollout simulator-assisted search budget, Stage A achieved a mean makespan of 25.597 versus 25.670 for the AttentionNet checkpoint, whereas Uniform+KL achieved 25.668. The pre-specified Uniform+KL–AttentionNet paired difference was −0.002 (95% stratified bootstrap CI: −0.157 to 0.150), providing no evidence of a statistically significant overall advantage. Stage A used approximately 25 thousand network parameters and required 1.495 s per instance compared with 2.362 s for AttentionNet in this evaluation environment. These results support competitive solution quality and a substantially smaller, faster model under the shared test protocol, not a definitive superiority claim.

## 9. التوصية

اعتمد **Stage A + Guided–Scarcity K=4** كأفضل مرجع وصفي ضمن النماذج الثلاثة، مع توضيح أن المقارنة الأساسية المحددة قبل التجربة كانت Uniform+KL ضد AttentionNet ولم تحقق تفوقاً. الأفضل حالياً تثبيت الإعدادات، كتابة البحث على أساس **كفاءة المعلمات وزمن التخطيط مع جودة حلول قريبة**، وإضافة تجارب Robustness وعدة بذور مستقلة عند الضرورة، دون إعادة ضبط الخوارزمية بناءً على نتائج الاختبار التأكيدي.
