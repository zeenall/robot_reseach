# رفع مستودع Paper 2 إلى GitHub

تم تجهيز الملفات كلها، لكن حساب GitHub المربوط بـChatGPT (`zeenall`) لا يظهر في الاتصال الحالي أي مستودعات أو صلاحية لإنشاء مستودع. **لذلك لم يتم نشر أي ملف على GitHub حتى الآن.**

## الطريقة الأسهل عبر GitHub CLI

1. ثبّت Git وGitHub CLI على Windows، ثم شغّل `gh auth login` لتسجيل الدخول إلى حسابك.
2. فك ضغط `KAN-HMRTA-GitHub-Ready.zip`.
3. داخل مجلد `KAN-HMRTA-Reproducibility` افتح PowerShell وشغّل `./PUBLISH_TO_GITHUB.ps1`.
4. سيتكوّن مستودع **عام** باسم `https://github.com/zeenall/KAN-HMRTA-Reproducibility` ويرفع الملفات دفعة واحدة. لا تشغّل السكربت إذا كان لديك مستودع يحمل الاسم نفسه وتحتوي داخله ملفات مهمة.

بدلاً من ذلك، أنشئ مستودعاً فارغاً من https://github.com/new باسم `KAN-HMRTA-Reproducibility` وأتِح ربطه بإضافة GitHub حتى يمكن إكمال الرفع من داخل ChatGPT عند ظهور صلاحية الكتابة.

## التحقق

```bash
python scripts/verify_release.py
```

### النسخة الخفيفة

إذا تعذر تنزيل النسخة الكاملة بسبب الحجم، استخدم ملف `KAN-HMRTA-GitHub-Lite.zip`، ثم بعد فك الضغط وتشغيل المتطلبات:

```bash
python scripts/download_official_attentionnet.py
python benchmark/run_direct_runtime_sampling10.py --mixed-per-mode 1 --sadcher-count 1 --out outputs/smoke
```

النسخة الخفيفة لا تتضمن أوزان الـAblations الاختيارية ولا النسخة المصدرة الكبيرة لوزن AttentionNet؛ ستحمّل أوزان AttentionNet مباشرة من المستودع الرسمي مع التحقق من البصمة.

**ملحوظة:** نشر الملف لا يعني منح ترخيص مفتوح تلقائياً لكودنا. راجع `CODE_PROVENANCE.md` واتفق مع بقية المؤلفين على الترخيص إذا رغبت أن يستخدم الآخرون الكود قانونياً.
