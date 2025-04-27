# 原理

一般我们使用[chartmuseum](https://chartmuseum.com/) 或者 [harbor](https://goharbor.io/docs/1.10/working-with-projects/working-with-images/managing-helm-charts/) 来存放和拉取helm chart的制品。

但实际上 helm 拉取chart的协议很简单：

- `GET /index.yaml` - retrieved when you run `helm repo add chartmuseum http://localhost:8080/`
- `GET /charts/mychart-0.1.0.tgz` - retrieved when you run `helm install chartmuseum/mychart`
- `GET /charts/mychart-0.1.0.tgz.prov` - retrieved when you run `helm install` with the `--verify` flag

也就是说只要提供一个静态文件服务器，实现 `/index.yaml` 和 `/charts/mychart-0.1.0.tgz` 的接口就够了，基于这样的原理我们可以使用 github pages 作为这样一个服务提供方。使用github actions 来更新chart 和 index.yaml 静态文件。

# Let's do it

1. 在存放源代码的 repo A中准备 chart 构建的文件夹，一般是 /chart 目录；
2. 创建另一个 github项目，专门用来存放构建的chart制品和index.yaml，一般**项目名**就叫 `charts`；
3. 由于需要在 repo A 的github action中往另一个项目里推送代码，为了更加安全地使用，需要创建一个 GitHub App，并使用该 app 来获取仓库的token。在右上角 Settings - Third-party Access /Github Apps -  New Github App；
	![[new github app.jpg]]

4. 设置基本信息Homepage URL 为 charts的github page地址，并生成一个`private key`，复制下来，待会会用到。设置权限可以访问 repo charts；

	![[repository access.jpg]]
	![[homepage  url.jpg]]

5. 在源码的repo的项目中，在 `Secrets and variables` 里添加 APP_ID 和 APP_PRIVATE_KEY。APP_ID可以在github app的页面找到，private key 就是刚才复制的key；
	![[repository variables.jpg]]
6. 在github actions workflow 中加上如下的代码，使用 action [create-github-app-token](https://github.com/actions/create-github-app-token) 获取token

```yaml
  publish-chart:
    if: startsWith(github.ref, 'refs/tags/v')
    env:
      HELM_CHARTS_DIR: charts
      HELM_CHART_NAME: jenkins
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Install Helm
        uses: azure/setup-helm@v3
      - name: Get the version
        id: get_version
        run: |
          VERSION=${GITHUB_REF#refs/tags/}
          echo "VERSION=${VERSION}" >> $GITHUB_OUTPUT

      - name: Tag helm chart image
        run: |
          image_tag=${{ steps.get_version.outputs.VERSION }}
          chart_version=${{ steps.get_version.outputs.VERSION }}
          sed -i "s/latest/${image_tag}/g" $HELM_CHARTS_DIR/values.yaml
          
          chart_smever=${chart_version#"v"}
          sed -i "s/0.1.0/${chart_smever}/g" $HELM_CHARTS_DIR/Chart.yaml

      - uses: getsentry/action-github-app-token@v2
        id: get_app_token
        with:
            app_id: ${{ secrets.APP_ID }}
            private_key: ${{ secrets.APP_PRIVATE_KEY }}
            
      - name: Sync Chart Repo
        run: |
          git config --global user.email "amamba[bot]@users.noreply.github.com"
          git config --global user.name "amamba[bot]"
          git clone https://x-access-token:${{ steps.get_app_token.outputs.token }}@github.com/amamba-io/charts.git amamba-charts
          helm package $HELM_CHARTS_DIR --destination ./amamba-charts/docs/
          helm repo index --url https://amamba-io.github.io/charts ./amamba-charts/docs/
          cd amamba-charts/
          git add docs/
          chart_version=${{ steps.get_version.outputs.VERSION }}
          git commit -m "update jenkins chart ${chart_version}"
          git push https://x-access-token:${{ steps.get_app_token.outputs.token }}@github.com/amamba-io/charts.git
```

7. 配置 repo charts ，在 Pages 页面选择source为 `Deploy from a branch` 并且选择 Branch 为 main 和 /docs 目录
	![[github pages source.png]]

## How to use

```
helm repo add chartrepo https://org-name.github.io/charts/
helm repo update chartrepo
```

# 参考

- [Helm | Chart Releaser Action to Automate GitHub Page Charts](https://helm.sh/docs/howto/chart_releaser_action/)

