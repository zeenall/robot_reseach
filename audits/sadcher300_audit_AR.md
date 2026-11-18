# Paper 2 — تدقيق نتائج التجربة 33: SADCHER-Generated Cross-Dataset Test (n=300)

**النتيجة الأساسية:** في مجموعة 300 مسألة مولّدة من مولّد SADCHER الأصلي (صفر قيود أسبقية)، تَفوَّق النظام Stage A + Guided–Scarcity Search K=4 على نظام AttentionNet weights + نفس عائلة البحث K=4 في متوسط Makespan داخل محاكي HeteroMRTA المشترك، بفارق زوجي −0.744111 وحدة HeteroMRTA (95% bootstrap CI كما ورد في الملف: [−0.931080, −0.559521]؛ إعادة حساب مستقلة بنحو 30,000 إعادة سحب: [−0.930193, −0.560961]). هذه النتيجة لا تثبت تفوق KAN وحده، أو التكافؤ بين المحاكيين، أو التفوق على خط الأنابيب الأصلي الكامل للباحثين الذي يتضمن Constrained Flashforward.

## 1. مصدر البيانات وبروتوكول التقييم

- الملف المدقَّق: `results (33).zip`، مجلد `PAPER2_SADCHER_CROSSDATASET_300`.
- عدد الحالات: 300 حالة ثابتة بذورها `2026121200 + i×1009`، مع `i=0..299`.
- كل حالة: 20 مهمة، و15 روبوتاً، و5 قدرات، ومن دون Precedence؛ نقطة البداية والنهاية مشتركة.
- تم توليد المسائل من `Sadcher_official_repository/data_generation/problem_generator.py` (بصمة SHA-256 تبدأ بـ`4a885558d682...`). هذا ليس تقييم المحاكي الأصلي لـSADCHER ولا المجموعة الرسمية لمقالته.
- التقييم لجميع النماذج نفذ داخل `HeteroMRTA TaskEnv` نفسه (إحداثيات ÷100؛ مدد تنفيذ ÷20). جميع الأوزان مجمدة؛ دون تدريب أو اختيار بذرة من نتائج هذه المجموعة.
- المقارنة الأساسية المسبقة بحسب بروتوكول التجربة: `stage_a - attention_original` عند K=4.
- K=1 هو المرشح الأول من عمليات البحث الأربع عند K=4؛ ميزانية K=4 تتضمن guided greedy، guided sample، scarcity greedy، scarcity sample. التوجيه مضبوط بمعاملات سابقة مختلفة لكل نموذج: Stage A `α=1.1, β=0.8`، AttentionNet `α=0.7, β=0.4`. يجب التصريح بذلك، وتفسير الفرق على مستوى النظام بعد الضبط وليس الشبكة العصبية منفردة.

## 2. متوسط النتائج

| النموذج | K | Mean Makespan | SD | Search elapsed s | Waiting time | Success |
|---|---:|---:|---:|---:|---:|---:|
| Stage A | 1 | 22.688179 | 3.266183 | 0.254890 | 1.570140 | 300/300 |
| AttentionNet weights | 1 | 23.233150 | 3.149705 | 0.435951 | 1.458932 | 300/300 |
| Stage A | 4 | **21.268011** | 2.997313 | **1.191474** | 1.327952 | 300/300 |
| AttentionNet weights | 4 | 22.012122 | 2.995233 | 1.919666 | 1.205316 | 300/300 |

**Relative mean-Makespan reduction:** 2.35% عند K=1، و3.38% عند K=4، محسوبة من فرق المتوسطين/متوسط AttentionNet. **Relative search-time reduction:** 41.53% عند K=1، و37.93% عند K=4 على بيئة الاختبار المسجَّلة. زمن البحث ليس قياساً معزولاً للشبكة أو ضمانة مقارنة على عتاد آخر.

## 3. فترات الثقة والتحليل الزوجي المستقل

| Metric | Mean difference (Stage A − Attention) | Original 95% bootstrap CI | Independent 95% CI (30k) | Stage A wins / Attention wins / ties |
|---|---:|---:|---:|---:|
| K=1 | −0.544972 | [−0.795144, −0.292090] | [−0.791438, −0.296199] | 180 / 118 / 2 |
| **K=4, primary** | **−0.744111** | **[−0.931080, −0.559521]** | **[−0.930193, −0.560961]** | **200 / 98 / 2** |

فترة الثقة الأساسية بالكامل تحت الصفر. كفحص تكميلي، اختبار الإشارة الزوجي ثنائي الطرف للـK=4 يعطي `p≈3.47×10⁻⁹` (بعد استبعاد التعادلات)، لكن اختبار Bootstrap المخطط له أساس الاستنتاج. لا تُفسّر الدلالة الإحصائية بأنها تفوق شامل على كل توزيعات MRTA.

## 4. فحص سلامة الملفات

1. اكتمل **1,200 سجل** = 300 حالة × نموذجين × وضعي بحث؛ كل سجل يشير إلى نجاح واكتمال candidate rollouts وإلى `replay_verified=1`.
2. عدد ملفات الخطط **1,200**. تم مطابقة كل ملف مع سجل `case_index/model/method/K`، وبصمة بيانات الحالة وبصمة مسار القرارات؛ أُعيد حساب بصمات جميع المسارات الـ1,200 من محتواها وتطابقت.
3. فحص 300 ملف حالات أثبت أبعاد القدرات والمتطلبات، و`n_precedence=0`، وتطابق مواقع الانطلاق والعودة، وعدم تكرار بصمات المسائل.
4. تطابقت SHA-256 لكل من أوزان Stage A، أوزان AttentionNet المنشورة، مولّد SADCHER، ملف `TaskEnv`، وكود التقييم مع قيم بروتوكول `protocol_locked.json`.
5. K=4 لم يكن أسوأ من K=1 في أي حالة لنفس النموذج؛ وحقق تخفيضاً فردياً في Makespan في 213 حالة لـStage A و217 حالة لـAttentionNet. **لم يُعَد تشغيل كل الـ1,200 جدول داخل المحاكي بشكل مستقل في عملية التدقيق هذه**؛ الاعتماد على سجل التنفيذ الأصلي الذي يعرض تحقق الإعادة، مع تدقيق ملفات القرارات والبصمات.

## 5. أهم تفسير: توزيع SADCHER المولَّد يغلب عليه تعدد المهارات

عبر الحالات الـ300، احتوت كل مسألة وسطياً على **16.14 مهمة متعددة المهارات من أصل 20** (نحو **80.7%** من المهام)، بمدى 10–20، وعلى **12.15 روبوتاً متعدد المهارات من أصل 15** تقريباً. لا توجد حالة كلها مهام one-hot. وبما أن الاختبارات السابقة أظهرت قوة Stage A نسبياً على نوع Multi-hot، فقد يكون اختلاف توزيع قدرات المهام أحد أسباب زيادة الفارق في مجموعة SADCHER مقارنة بمجموعة 501 حالة المتوازنة بين One-hot وMulti-hot وHard. هذا تفسير احتمالي وليس سبباً مُثبتاً تجريبياً.

عند تقسيم المسائل حسب عدد المهام متعددة المهارات، كان فارق Stage A − AttentionNet عند K=4 سالباً في كل مجموعات التقسيم الاستكشافية: (10–15): −0.773؛ (16): −0.546؛ (17): −0.797؛ (18–20): −0.832. لا تُعرض هذه النتائج على أنها اختبار فرضيات مستقل مصحح للتعدد.

## 6. المقارنة مع تجربة 501 حالة السابقة

في اختبار 501 حالة السابق كان الفرق عند K=4 حوالي −0.0730، وفترة الثقة شملت الصفر. هذه المجموعة الجديدة ذات مصدر توليد وتوزيع مهارات مختلف أظهرت −0.7441 بفترة ثقة تستبعد الصفر. لا يصح مقارنة متوسط Makespan المطلق (25.6 مقابل 21.3 مثلاً) لتقييم تحسن النموذج عبر الاختبارين: النماذج لم تتغير، وتوزيعات صعوبة المسائل مختلفة.

## 7. الادعاء الدقيق المناسب للورقة

> On 300 previously unseen instances generated by the published SADCHER problem generator with zero precedence constraints, both frozen policies were evaluated in the same unchanged HeteroMRTA simulator under an identical four-rollout search budget. The Stage A KAN-based system achieved a mean makespan of 21.268 compared with 22.012 for the published AttentionNet weights under our shared Guided–Scarcity search protocol (paired mean difference −0.744; 95% bootstrap CI [−0.931, −0.560]). Both achieved 100% completion. The Stage A system also recorded 37.9% lower mean search elapsed time in this execution environment. These findings support cross-dataset generalization on this generator's distribution; they do not establish cross-simulator equivalence or superiority over the published authors' complete Constrained Flashforward pipeline.

الورقة يمكن أن تصف هذه التجربة **Cross-Dataset Evaluation Using SADCHER-Generated Instances**، لا Cross-Simulator Transfer. ولا بد أن توضّح أيضاً أن هذه المجموعة ذات نسبة كبيرة من المهام متعددة المهارات؛ لذا لا تُغني عن التقييم المتوازن السابق ولا عن تجربة أحجام روبوتات/مهام جديدة إذا أردنا ادعاء scalability.

## 8. مصادر الكود

- مولّد المسائل المنشور: https://github.com/jakbichler/Sadcher
- بيئة المصدر الأصلية: https://github.com/marmotlab/HeteroMRTA
- بياناتنا التفصيلية والمصدر المباشر للاستنتاجات: ملف المستخدم `results (33).zip`، ولا سيما `evaluation_per_case.csv`, `evaluation_summary.csv`, `paired_statistics.csv`, `protocol_locked.json`, `scientific_scope.json`, وملفات `cases/` و`plans/`.
