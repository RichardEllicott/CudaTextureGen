// using Godot;
// using System;

// public partial class PythonBridge : Node
// {
// }

using Godot;
using System.Diagnostics;
using System.Text;

[Tool]
[GlobalClass]
public partial class PythonBridge : Node
{
    private Process python;

    public override void _Ready()
    {
        python = new Process();
        python.StartInfo.FileName = "python3"; // or full path to python.exe
        python.StartInfo.Arguments = "projects/python_test/python_test.py";
        python.StartInfo.UseShellExecute = false;
        python.StartInfo.RedirectStandardInput = true;
        python.StartInfo.RedirectStandardOutput = true;
        python.StartInfo.RedirectStandardError = true;
        python.StartInfo.CreateNoWindow = true;

        python.Start();
    }

    public void SendLine(string line)
    {
        if (python == null || python.HasExited)
            return;

        python.StandardInput.WriteLine(line);
        python.StandardInput.Flush();
    }

    public string ReadLine()
    {
        if (python == null || python.HasExited)
            return "";

        return python.StandardOutput.ReadLine();
    }

    public bool IsRunning()
    {
        return python != null && !python.HasExited;
    }

    public override void _ExitTree()
    {
        if (python != null && !python.HasExited)
            python.Kill();
    }
}

