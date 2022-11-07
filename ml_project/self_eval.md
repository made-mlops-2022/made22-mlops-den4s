## Критерии

### Сделано (возможно частично)

1. описание в пулл-реквесте того, что сделано и для чего (`1` балл из `1`)

2. проведена самооценка, по каждому пункту расписано выполнен критерий или нет и на сколько баллов (`1` балл из `1`)

3. выполнено EDA: ноутбук в папке `notebooks` (`1` балл из `1`), возможно с моделью

4. использован скрипт, генерирующий отчет, закоммичены и скрипт и отчет (`+1` балл)

5. написана функция `train_pipeline.py` для тренировки модели, вызов оформлен как утилита командной строки, инструкцию по запуску записана в `README.md` (`3` балла из `3`)

6. написана функция `eval_pipeline.py` (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в `README.md` (`3` балла из `3`)

7. проект имеет модульную структуру (`2` балла из `2`)

8. использованы логгеры (`2` балла из `2`)

9. написаны тесты на отдельные модули и на прогон обучения и оценки (`3` балла из `3`)

10. для тестов генерируются синтетические данные, приближенные к реальным (`2` балла из `2`)

11. обучение модели конфигурируется с помощью конфигов в json или yaml, закоммичены как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split - _в этом проекте одна стратегия_, preprocessing) (`2` балла из `3`)

12. используются датаклассы для сущностей из конфига, а не голые dict (`2` балла из `2`)

13. написан и протестирован кастомный трансформер (`3` балла из `3`)

14. в проекте зафиксированы все зависимости (`1` балл из `1`)

### Не сделано

1. настроен CI для прогона тестов, линтера на основе github actions (`0` баллов из `3`)

### Сумма
`26 (+1)` из `29`
