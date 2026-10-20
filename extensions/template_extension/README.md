# Falcon Template Extension

A template extension for creating custom Falcon processors and data types.

* **DummyData**: Dummy data type with a string named `value`.
* **DummyWriter**: Generates `DummyData` periodically.
* **DummyReader**: Consumes `DummyData` and prints the `value` to the console.

A simple example pipeline would be:
```yaml
processors:
    my_writer:
        class: DummyWriter
        options:
            freq: 6
            message: "hello"

    my_reader:
        class: DummyReader

connections:
    - my_writer.output = my_reader.input
```


Duplicate and modify this extension to make it your own.
