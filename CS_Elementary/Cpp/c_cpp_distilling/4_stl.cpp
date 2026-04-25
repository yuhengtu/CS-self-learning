// reacall Cpp-number-array std::vector, dynamic length, no decay to pointer
    vector<int> v = {0, 1, 2};
    cout << v[0] << ' ' << v[1] << ' ' << v[2] << endl;
    cout << v.size() << endl;
    vector<vector<int>> v2d(2, vector<int>(3, 0));
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << v2d[i][j] << " ";
        }
        cout << endl;
    }
    printf("\n");