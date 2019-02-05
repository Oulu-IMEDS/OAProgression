import pretrainedmodels
from torch import nn


class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PretrainedModel(nn.Module):
    def __init__(self, backbone, drop, ncls, pretrained=True):
        super().__init__()
        if pretrained:
            model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained='imagenet')
        else:
            model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained=None)
        self.encoder = list(model.children())[:-2]

        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)

        if drop > 0:
            self.fc = nn.Sequential(FCViewer(),
                                    nn.Dropout(drop),
                                    nn.Linear(model.last_linear.in_features, ncls))
        else:
            self.fc = nn.Sequential(
                FCViewer(),
                nn.Linear(model.last_linear.in_features, ncls)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x


class KneeNet(nn.Module):
    """
    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """

    def __init__(self, backbone_net, drop, pretrained=True):
        super(KneeNet, self).__init__()
        backbone = PretrainedModel(backbone_net, 1, 1, pretrained)

        self.features = backbone.encoder

        # 4 KL-grades
        self.classifier_kl = nn.Sequential(nn.Dropout(p=drop),
                                           nn.Linear(backbone.fc[-1].in_features, 4))
        # 3 progression sub-types
        self.classifier_prog = nn.Sequential(nn.Dropout(p=drop),
                                             nn.Linear(backbone.fc[-1].in_features, 3))

    def forward(self, x):
        o = self.features(x)
        feats = o.view(o.size(0), -1)
        return self.classifier_kl(feats), self.classifier_prog(feats)
